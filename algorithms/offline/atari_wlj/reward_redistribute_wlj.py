"""Reward redistribution networks and DARTS-style bilevel meta-gradient.

Ported from the official DTRD repository (``reward_redistribute.py`` and the
``TrajectoryDataset`` in ``run_atari.py``) so the reproduction matches the paper
"Towards Long-delayed Sparsity" (Zhu et al., IJCAI 2023) mechanism exactly.

Adaptations for CORL:
- ``GradientApproximate`` no longer requires the policy model to carry a
  ``.decay`` attribute or a ``.copy_model`` method. Instead the set of
  weight-decayed parameter names is passed in explicitly (see
  :func:`decay_param_names`), and the unrolled copy ``new_model`` is created by
  the caller (see :func:`clone_gpt`). This keeps CORL's shared ``model_atari.py``
  untouched.
"""

import copy

import numpy as np
import torch
import torch.nn as nn


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class RedistributeConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DiscreteRedistributeNetwork(nn.Module):
    """Discrete reward redistribution via gumbel-softmax over reward categories."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.state_encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
            nn.Flatten(), nn.Linear(3136, config.n_embd), nn.Tanh(),
        )
        self.action_embeddings = nn.Sequential(
            nn.Embedding(config.action_dim, config.n_embd), nn.Tanh()
        )
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2 * config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, 16 * config.n_embd)
        self.fc3 = nn.Linear(16 * config.n_embd, config.reward_category_num)
        self.reward_vector = torch.tensor(
            np.array(config.reward_vector), dtype=torch.float32
        ).to(config.device)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight, gain=np.sqrt(2))
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, states, actions):
        state_embeddings = self.state_encoder(
            states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous()
        )
        state_embeddings = state_embeddings.reshape(
            states.shape[0], states.shape[1], self.config.n_embd
        )
        action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1))
        token_embeddings = torch.cat([state_embeddings, action_embeddings], -1)
        x = self.relu(self.fc1(token_embeddings))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_redistribute(self, states, actions):
        r = self.forward(states, actions)
        one_hot = nn.functional.gumbel_softmax(r, tau=1.0, hard=True, dim=-1)
        redistribute = (one_hot * self.reward_vector).sum(-1)
        return redistribute


class ContinuousRedistributeNetwork(nn.Module):
    """Continuous reward redistribution, rescaled into [min_reward, max_reward]."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.state_encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
            nn.Flatten(), nn.Linear(3136, config.n_embd), nn.Tanh(),
        )
        self.action_embeddings = nn.Sequential(
            nn.Embedding(config.action_dim, config.n_embd), nn.Tanh()
        )
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2 * config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, 16 * config.n_embd)
        self.fc3 = nn.Linear(16 * config.n_embd, 1)
        self.reward_range = config.reward_range
        self.redistribute_activate_func = config.redistribute_activate_func
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight, gain=np.sqrt(2))
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, states, actions):
        state_embeddings = self.state_encoder(
            states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous()
        )
        state_embeddings = state_embeddings.reshape(
            states.shape[0], states.shape[1], self.config.n_embd
        )
        action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1))
        token_embeddings = torch.cat([state_embeddings, action_embeddings], -1)
        x = self.relu(self.fc1(token_embeddings))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        if self.redistribute_activate_func == "tanh":
            x = torch.tanh(x)
        else:
            x = torch.sigmoid(x)
        return x

    def get_redistribute(self, state, action):
        r = self.forward(state, action)
        r = r.squeeze(-1)
        rng = self.reward_range
        if self.redistribute_activate_func == "tanh":
            r = (rng["max_reward"] - rng["min_reward"]) * (r + 1.0) / 2.0 + rng["min_reward"]
        else:
            r = (rng["max_reward"] - rng["min_reward"]) * r + rng["min_reward"]
        return r


def clone_gpt(model):
    """Return a deep-copied policy model for the DARTS one-step unroll (φ').

    The official DTRD relies on ``model.copy_model()``; CORL's GPT has no such
    method, so we deep-copy. The copy shares no storage with the original.
    """
    return copy.deepcopy(model)


def decay_param_names(model):
    """Replicate GPT.configure_optimizers' decay/no-decay partitioning and
    return the SET of fully-qualified parameter names that receive weight decay.

    ``GradientApproximate.compute_unrolled_model`` needs to know which params
    get weight decay in the unroll step; the official code reads ``model.decay``
    for this. We recompute the same set here to avoid touching model_atari.py.
    """
    decay = set()
    no_decay = set()
    whitelist = (nn.Linear, nn.Conv2d)
    blacklist = (nn.LayerNorm, nn.Embedding)
    for mn, m in model.named_modules():
        for pn, _ in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist):
                no_decay.add(fpn)
    no_decay.add("pos_emb")
    no_decay.add("global_pos_emb")
    return decay


class GradientApproximate:
    """DARTS-style second-order bilevel meta-gradient for the redistribution net.

    Ported from the official DTRD ``reward_redistribute.py``. The only CORL
    adaptation: the set of weight-decayed policy parameter names is passed in as
    ``decay_names`` (see :func:`decay_param_names`) instead of being read from a
    ``model.decay`` attribute.
    """

    def __init__(self, device, new_model, decay_names, eps=0.01, weight_decay=0.1):
        self.device = device
        self.new_model = new_model
        self.decay_names = decay_names
        self.eps = eps
        self.weight_decay = weight_decay

    def redistribute_step(self, model, redistribute, redistribute_sum_square, policy_optimizer,
                          x_t, y_t, r_t, t_t, x_v, y_v, r_v, t_v, eta):
        # model(φ') : one-step unrolled policy copy
        self.compute_unrolled_model(model, policy_optimizer, x_t, y_t, r_t, t_t, eta)
        # L_val(θ, φ')
        val_logits, val_loss = self.new_model(x_v, y_v, y_v, r_v, t_v)
        self.new_model.zero_grad()
        redistribute.zero_grad()
        val_loss.backward(retain_graph=True)
        # dφ' = d(L_val(θ, φ')) / d(φ')
        vector = [v.grad if v.grad is not None else torch.zeros_like(v)
                  for v in self.new_model.parameters()]
        # d(L_val(θ, φ')) / dθ
        d_Lval_theta = [v.grad.clone().detach() if v.grad is not None else torch.zeros_like(v)
                        for v in redistribute.parameters()]

        # Second-order approximation via finite-difference Hessian-vector product
        appro_vector_product = self.hessian_vector_product(
            model, redistribute, vector, x_t, y_t, r_t, t_t
        )

        # First-order regularizer term: d[lambda * sum(redistribute)^2] / dθ
        redistribute.zero_grad()
        redistribute_sum_square.backward()
        with torch.no_grad():
            for ap, dl, appro in zip(redistribute.parameters(), d_Lval_theta, appro_vector_product):
                ap.grad = ap.grad + (dl - eta * appro)
        return val_loss, val_logits

    def compute_unrolled_model(self, model, policy_optimizer, x_t, y_t, r_t, t_t, eta):
        # L_train(θ, φ)
        _, train_loss = model(x_t, y_t, y_t, r_t, t_t)
        model.zero_grad()
        # dL_train(θ, φ) / dφ
        d_phi = torch.autograd.grad(
            train_loss, model.parameters(), allow_unused=True, retain_graph=True,
            grad_outputs=torch.ones_like(train_loss),
        )
        with torch.no_grad():
            for (name, w), vw, g in zip(model.named_parameters(), self.new_model.parameters(), d_phi):
                if g is None:
                    g = torch.zeros_like(w)
                weight_decay = self.weight_decay if name in self.decay_names else 0.0
                # φ' = φ - η * (dL_train/dφ + φ * φ_decay)
                vw.copy_(w - eta * (g + weight_decay * w))

    def hessian_vector_product(self, model, redistribute, vector, x_t, y_t, r_t, t_t):
        R = self.eps / _concat(vector).norm()
        # φ+ = φ + eps * dφ'
        with torch.no_grad():
            for p, v in zip(model.parameters(), vector):
                p += R * v
        _, loss = model(x_t, y_t, y_t, r_t, t_t)
        grads_p = torch.autograd.grad(
            loss, redistribute.parameters(), allow_unused=True, retain_graph=True,
            grad_outputs=torch.ones_like(loss),
        )
        # φ- = φ - 2 * eps * dφ'
        with torch.no_grad():
            for p, v in zip(model.parameters(), vector):
                p -= 2.0 * R * v
        _, loss = model(x_t, y_t, y_t, r_t, t_t)
        grads_n = torch.autograd.grad(
            loss, redistribute.parameters(), allow_unused=True, retain_graph=True,
            grad_outputs=torch.ones_like(loss),
        )
        # restore φ
        with torch.no_grad():
            for p, v in zip(model.parameters(), vector):
                p += R * v
        return [(gp - gn) / (2.0 * R) for gp, gn in zip(grads_p, grads_n)]


class TrajectoryDataset:
    """Per-trajectory bookkeeping for reward redistribution.

    Ported from the official DTRD ``run_atari.py`` TrajectoryDataset. Stores the
    head (first-step) sparse rtg and length of each trajectory, and computes the
    redistributed return-to-go for a batch of windows.

    ``discrete_redistribute`` and ``trajectory_lamb`` are passed in (the official
    code read them from a module-level ``args``).
    """

    def __init__(self, states, actions, rtgs, done_idxs, discrete_redistribute, trajectory_lamb):
        self.discrete_redistribute = discrete_redistribute
        self.trajectory_lamb = trajectory_lamb
        self.head_rtgs = []
        self.traj_len = []
        start = 0
        for done in done_idxs:
            done = int(done)
            self.head_rtgs.append(rtgs[start])
            self.traj_len.append(done - start)
            start = done

    def get_redistribute_rtgs_local(self, states, actions, timesteps, indexes,
                                    redistribute_network, device, calculate_sum_square=False):
        batch_size, context_length = states.shape[0], states.shape[1]
        step_redis_reward = redistribute_network.get_redistribute(states, actions).reshape(
            batch_size, 1, context_length
        )
        head_rtgs = self.get_head_rtgs_from_index(indexes).to(device)
        traj_len = self.get_traj_length_from_index(indexes).to(device)
        head_timestep = timesteps.reshape(timesteps.shape[0], 1).to(dtype=torch.float32)
        head_redis_rewards = (
            head_rtgs * head_timestep / traj_len
            if not self.discrete_redistribute
            else torch.round(head_rtgs * head_timestep / traj_len)
        ).repeat(1, context_length).unsqueeze(-1)
        trajectory_redis_rtgs = torch.matmul(
            step_redis_reward,
            torch.tensor(
                np.triu(torch.ones((context_length, context_length))) - np.eye(context_length),
                dtype=torch.float32,
            ).unsqueeze(0).repeat(batch_size, 1, 1).to(device),
        ).reshape(batch_size, context_length, 1)

        trajectory_redis_rtgs = trajectory_redis_rtgs + head_redis_rewards
        if calculate_sum_square:
            redis_sum_square = (
                torch.sum(step_redis_reward, dim=-1) * traj_len / context_length - head_rtgs
            ).pow(2).sum() * self.trajectory_lamb / batch_size
        else:
            redis_sum_square = torch.zeros(1).to(device)
        return trajectory_redis_rtgs, redis_sum_square

    def get_head_rtgs_from_index(self, indexes):
        head_indexes = indexes[:, 0, ]
        return torch.from_numpy(np.array(self.head_rtgs))[head_indexes]

    def get_traj_length_from_index(self, indexes):
        head_indexes = indexes[:, 0, ]
        return torch.from_numpy(np.array(self.traj_len, dtype=np.float32))[head_indexes]
