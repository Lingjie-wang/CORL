"""Bilevel DTRD trainer for CORL.

Ported from the official DTRD ``mingpt/trainer_atari.py`` (the ``ModelTrainer``
bilevel loop) and ``eval_atari.py`` (``play_real_game``). Adapted to CORL:
- reuses CORL's ``Env`` / ``Args`` (trainer_atari.py) and ``sample`` (utils.py)
  for evaluation — CORL's Env already matches the official one (sticky actions
  off, life-loss non-terminal in eval mode);
- the DARTS meta-gradient (``GradientApproximate``) receives the decay-param
  name set and an externally-cloned unrolled model, so CORL's GPT is untouched;
- adds wandb logging of train/val loss, redistribution loss and eval returns,
  which the paper's code did not have.
"""

import logging
import math

import numpy as np
import torch
import wandb
from torch.utils.data.dataloader import DataLoader

from algorithms.offline.atari_wlj.reward_redistribute_wlj import (
    GradientApproximate,
    clone_gpt,
    decay_param_names,
)
from algorithms.offline.atari_wlj.trainer_atari import Args, Env
from algorithms.offline.atari_wlj.utils import sample

logger = logging.getLogger(__name__)


class DTRDTrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 1e-3
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = True
    warmup_tokens = 512 * 20
    final_tokens = 260e9
    num_workers = 0
    seed = 123
    game = "Breakout"
    max_timestep = 1
    device = "cuda"
    context_length = 30
    # redistribution optim
    redistribute_learning_rate = 1e-3
    redistribute_step_size = 1000
    redistribute_gamma = 0.9
    # eval
    eval_episodes = 10
    eval_target_return = None
    max_eval_steps = 27000  # hard cap per episode (~108k frames / 4 action-repeat)
    ckpt_path = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DTRDModelTrainer:
    def __init__(self, model, redistribute, device, train_dataset, val_dataset,
                 train_trajectory_dataset, val_trajectory_dataset, config):
        self.config = config
        self.model = model
        self.new_model = clone_gpt(model).to(device)
        self.redistribute = redistribute
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_trajectory_dataset = train_trajectory_dataset
        self.val_trajectory_dataset = val_trajectory_dataset

        self.now_learning_rate = config.learning_rate
        self.tokens = 0
        self.global_step = 0
        self.best_return = -float("inf")

        self.model = self.model.to(device)
        self.optimizer = model.configure_optimizers(config)
        self.redistribute_optimizer = torch.optim.Adam(
            self.redistribute.parameters(), lr=config.redistribute_learning_rate
        )
        self.redistribute_scheduler = torch.optim.lr_scheduler.StepLR(
            self.redistribute_optimizer,
            step_size=config.redistribute_step_size,
            gamma=config.redistribute_gamma,
        )
        self.gradient_appro = GradientApproximate(
            device=self.device,
            new_model=self.new_model,
            decay_names=decay_param_names(model),
            eps=0.01,
            weight_decay=config.weight_decay,
        )

    def save_checkpoint(self):
        if self.config.ckpt_path is None:
            return
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(
            {
                "policy_network": raw_model.state_dict(),
                "redistribute_network": self.redistribute.state_dict(),
                "max_timestep": self.config.max_timestep,
            },
            self.config.ckpt_path,
        )

    def _lr_decay_step(self, y):
        """DT-style linear-warmup + cosine-decay applied to the policy optimizer.
        Mirrors the official per-step schedule; returns the current lr."""
        config = self.config
        if config.lr_decay:
            self.tokens += (y >= 0).sum()
            if self.tokens < config.warmup_tokens:
                lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
            else:
                progress = float(self.tokens - config.warmup_tokens) / float(
                    max(1, config.final_tokens - config.warmup_tokens)
                )
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lr = config.learning_rate * lr_mult
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        else:
            lr = config.learning_rate
        self.now_learning_rate = lr
        return lr

    def train(self):
        model, redistribute, config = self.model, self.redistribute, self.config
        train_traj = self.train_trajectory_dataset
        val_traj = self.val_trajectory_dataset

        def run_epoch(epoch):
            from tqdm import tqdm

            model.train()
            # Train drives the epoch: full dataset, shuffled. The official DTRD
            # ships equally-sized train/val npz and zips them; here we build one
            # dataset and hold out a small val split, so zipping directly would
            # truncate the epoch to the (much smaller) val length and, with
            # shuffle off, replay the same handful of windows every epoch. We
            # instead iterate the full shuffled train set and pull one val batch
            # per step from a cycling val iterator (standard DARTS bilevel setup).
            train_loader = DataLoader(
                self.train_dataset, shuffle=True, pin_memory=True,
                batch_size=config.batch_size, num_workers=config.num_workers,
                drop_last=True,
            )
            val_loader = DataLoader(
                self.val_dataset, shuffle=True, pin_memory=True,
                batch_size=config.batch_size, num_workers=config.num_workers,
                drop_last=True,
            )

            def cycle(loader):
                while True:
                    for batch in loader:
                        yield batch

            val_iter = cycle(val_loader)
            train_losses, val_losses = [], []
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for it, (x_t, y_t, r_t, t_t, traj_t) in pbar:
                x_v, y_v, r_v, t_v, traj_v = next(val_iter)
                x_t, y_t, r_t, t_t = (z.to(self.device) for z in (x_t, y_t, r_t, t_t))
                x_v, y_v, r_v, t_v = (z.to(self.device) for z in (x_v, y_v, r_v, t_v))

                # redistributed rtgs for the train batch (no regularizer)
                redistribute_reward_t, _ = train_traj.get_redistribute_rtgs_local(
                    states=x_t, actions=y_t, timesteps=t_t, indexes=traj_t,
                    redistribute_network=redistribute, device=self.device,
                    calculate_sum_square=False,
                )
                r_t_m = r_t - redistribute_reward_t.clone().detach()
                r_t = r_t - redistribute_reward_t

                # redistributed rtgs for the val batch (with trajectory regularizer)
                redistribute_reward_v, redistribute_sum_square = val_traj.get_redistribute_rtgs_local(
                    states=x_v, actions=y_v, timesteps=t_v, indexes=traj_v,
                    redistribute_network=redistribute, device=self.device,
                    calculate_sum_square=True,
                )
                r_v_m = r_v - redistribute_reward_v.clone().detach()
                r_v = r_v - redistribute_reward_v

                # Upper level: meta-update the redistribution network
                eta = self.now_learning_rate
                val_loss, _ = self.gradient_appro.redistribute_step(
                    model, redistribute, redistribute_sum_square, self.optimizer,
                    x_t, y_t, r_t, t_t, x_v, y_v, r_v, t_v, eta,
                )
                val_losses.append(val_loss.item())
                reg_loss = float(redistribute_sum_square.item())  # trajectory_lamb * sum_square
                torch.nn.utils.clip_grad_norm_(redistribute.parameters(), config.grad_norm_clip)
                self.redistribute_optimizer.step()
                self.redistribute_scheduler.step(None)

                # Lower level, step 1: policy update on the train batch
                _, loss = model(x_t, y_t, y_t, r_t_m, t_t)
                loss = loss.mean()
                policy_train_loss = loss.item()
                train_losses.append(policy_train_loss)
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()
                self._lr_decay_step(y_t)

                # Lower level, step 2: policy update on the val batch
                _, loss = model(x_v, y_v, y_v, r_v_m, t_v)
                loss = loss.mean()
                policy_val_loss = loss.item()
                train_losses.append(policy_val_loss)
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()
                lr = self._lr_decay_step(y_v)

                wandb.log(
                    {
                        # policy (lower level): CE on the train and val minibatches
                        "policy/train_loss": policy_train_loss,
                        "policy/val_loss": policy_val_loss,
                        # redistribution net (upper level): meta val loss + regularizer
                        "redistribute/meta_val_loss": val_loss.item(),
                        "redistribute/reg_loss": reg_loss,
                        # kept for backward compatibility with earlier runs
                        "train_loss": policy_train_loss,
                        "val_loss": val_loss.item(),
                        "learning_rate": lr,
                        "epoch": epoch,
                    },
                    step=self.global_step,
                )
                self.global_step += 1
                if it % 100 == 0:
                    pbar.set_description(
                        f"epoch {epoch + 1} iter {it}: policy loss {policy_train_loss:.5f}. lr {lr:e}"
                    )

            logger.info("epoch %d mean policy loss %f mean meta-val loss %f", epoch + 1,
                        float(np.mean(train_losses)), float(np.mean(val_losses)))

        for epoch in range(config.max_epochs):
            run_epoch(epoch)
            if config.eval_episodes > 0:
                self.best_return = self.evaluate(epoch, self.best_return)

    @torch.no_grad()
    def _rollout_eval(self, decrement_rtg):
        """Run eval_episodes rollouts and return the mean env return.

        decrement_rtg=True  -> official DTRD conditioning: rtg starts at target
            and is decremented each step by the LEARNED redistribution reward
            (clipped >=0).
        decrement_rtg=False -> DT-style control: rtg is held CONSTANT at target
            for the whole episode (mimics how the plain DT-sparse baseline is
            conditioned). Comparing the two isolates whether the rtg-decrement
            mechanism is what suppresses DTRD's score.
        """
        config = self.config
        target_return = config.eval_target_return
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        args = Args(config.game.lower(), config.seed, self.device)
        env = Env(args)
        env.eval()

        T_rewards = []
        for _ in range(config.eval_episodes):
            state = env.reset()
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [float(target_return)]
            sampled_action = sample(
                raw_model, state, 1, temperature=1.0, sample=False, actions=None,
                rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(-1),
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device),
            )
            j = 0
            reward_sum = 0
            window = config.context_length
            prev_state = state
            all_states = state
            actions = []
            while True:
                action = sampled_action.cpu().numpy()[0, -1]
                actions += [sampled_action]
                state, reward, done = env.step(action)
                reward_sum += reward
                j += 1
                if done or j >= config.max_eval_steps:
                    T_rewards.append(float(reward_sum))
                    break
                state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                if decrement_rtg:
                    # official DTRD: rtg -= learned redistribution reward, clipped >=0
                    now_action = torch.tensor(action, dtype=torch.long).to(self.device).reshape(1, 1, 1)
                    redistribute_reward = self.redistribute.get_redistribute(prev_state, now_action).item()
                    rtgs += [max(0.0, rtgs[-1] - redistribute_reward)]
                else:
                    # DT-style control: hold rtg constant at target
                    rtgs += [rtgs[-1]]
                prev_state = state

                all_states = torch.cat([all_states, state], dim=0)[-window:]
                actions = actions[-window:]
                rtgs = rtgs[-window:]
                sampled_action = sample(
                    raw_model, all_states.unsqueeze(0), 1, temperature=1.0, sample=False,
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0),
                    rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(-1),
                    timesteps=(min(j, config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)),
                )
        env.close()
        return sum(T_rewards) / float(config.eval_episodes)

    @torch.no_grad()
    def evaluate(self, epoch, best_return):
        """Run BOTH eval modes each epoch to isolate the rtg-decrement effect:
        eval/return       = official DTRD (rtg decremented by redistribution)
        eval/return_dt    = DT-style control (rtg held constant at target)
        A large gap (dt >> return) means the rtg-decrement mechanism, not the
        trained policy, is what suppresses the score."""
        config = self.config
        self.model.train(False)
        self.redistribute.train(False)

        eval_return = self._rollout_eval(decrement_rtg=True)
        eval_return_dt = self._rollout_eval(decrement_rtg=False)
        best_return = max(best_return, eval_return)
        wandb.log(
            {
                "eval/return": eval_return,
                "eval/return_dt": eval_return_dt,
                "eval/best_return": best_return,
                "eval/target_return": float(config.eval_target_return),
                "epoch": epoch,
            },
            step=self.global_step,
        )
        self.global_step += 1
        logger.info("eval return (DTRD rtg-decrement): %f | eval return (DT constant-rtg): %f",
                    eval_return, eval_return_dt)
        self.model.train(True)
        self.redistribute.train(True)
        return best_return
