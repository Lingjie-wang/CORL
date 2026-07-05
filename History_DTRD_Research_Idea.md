# 基于历史信息改进 DTRD Reward Model 的思路

我认为这个想法值得尝试，而且相比引入多个网络和多个损失函数，这个方案更加简单、稳定，也更容易分析。

它只修改 DTRD 中的一个核心假设：

$$
\boxed{
\text{DTRD：当前状态动作决定代理奖励}
\quad\longrightarrow\quad
\text{改进：完整历史前缀决定代理奖励}
}
$$

具体来说，将原始 DTRD 中的：

$$
\hat r_t=f_\phi(s_t,a_t)
$$

修改为：

$$
\hat r_t=f_\phi(\tau_{1:t}),
$$

其中：

$$
\tau_{1:t}
=
(s_1,a_1,s_2,a_2,\ldots,s_t,a_t)
$$

表示从轨迹开始到当前时间步的完整历史前缀。

这个改进希望回答一个明确的研究问题：

> **DTRD 中的 Markov reward model 是否限制了奖励重分配？引入历史信息能否改善稀疏奖励场景下 Decision Transformer 的性能？**

---

# 一、这个改动针对 DTRD 的什么问题？

DTRD 原始的代理奖励模型为：

$$
\hat r_t=f_\phi(s_t,a_t),
\qquad
f_\phi:\mathcal S\times\mathcal A\rightarrow\mathbb R.
$$

这意味着，只要两个时间步具有相同的状态动作对：

$$
(s_t,a_t),
$$

无论它们之前经历了什么历史，reward model 都必须输出相同的代理奖励。

也就是说，DTRD 默认假设代理奖励是 Markov 的：

$$
\hat r_t
\perp
\tau_{1:t-1}
\mid
(s_t,a_t).
$$

但在某些稀疏奖励任务中，当前状态动作对可能无法完整表示任务进度。

例如，在“拿钥匙并开门”的任务中，智能体当前都位于门前，并执行相同的开门动作：

$$
(s_t,a_t)
=
(\text{位于门前},\text{开门}).
$$

但是可能存在两种不同的历史。

## 情况一：之前已经拿到钥匙

$$
\tau^{(1)}_{1:t}
=
(\text{探索},\text{拿到钥匙},\text{走到门前},\text{开门}).
$$

这个开门动作能够使任务成功，因此应该获得较高的代理奖励。

## 情况二：之前没有拿到钥匙

$$
\tau^{(2)}_{1:t}
=
(\text{探索},\text{错过钥匙},\text{走到门前},\text{开门}).
$$

这个开门动作可能完全无效，因此不应该获得较高奖励。

如果当前状态中没有明确包含“是否已经拿到钥匙”这一信息，那么原始 DTRD 的输入完全相同：

$$
(s_t,a_t).
$$

因此，原始 reward model 只能输出：

$$
f_\phi(s_t,a_t)
=
\text{相同的代理奖励}.
$$

而历史条件 reward model 可以输出：

$$
f_\phi(\tau^{(1)}_{1:t})
\neq
f_\phi(\tau^{(2)}_{1:t}).
$$

这就是引入历史信息可能有效的根本原因。

---

# 二、建议采用的模型形式

不建议直接将完整轨迹前缀展平后输入 MLP，而应该使用序列模型编码历史信息。

可以先定义历史表示：

$$
h_t
=
\operatorname{Encoder}_\phi
(s_1,a_1,\ldots,s_t,a_t),
$$

然后根据历史表示预测代理奖励：

$$
\hat r_t
=
g_\phi(h_t).
$$

合起来可以写成：

$$
\boxed{
\hat r_t
=
f_\phi(\tau_{1:t})
}
$$

其中 Encoder 可以选择：

- GRU；
- LSTM；
- causal Transformer。

## 1. GRU 版本

使用 GRU 时，可以写为：

$$
h_t
=
\operatorname{GRU}_\phi
\left(
h_{t-1},
[s_t,a_t]
\right),
$$

$$
\hat r_t
=
g_\phi(h_t).
$$

其中：

- $[s_t,a_t]$ 表示状态和动作的拼接；
- $h_{t-1}$ 保存之前的历史信息；
- $h_t$ 表示截至当前时间步的历史状态；
- $g_\phi$ 可以是一个简单的线性层或 MLP。

## 2. causal Transformer 版本

也可以使用因果 Transformer：

$$
h_t
=
\operatorname{CausalTransformer}_\phi
(s_1,a_1,\ldots,s_t,a_t),
$$

$$
\hat r_t
=
g_\phi(h_t).
$$

因果 Transformer 在预测时间步 $t$ 的奖励时，只允许看到：

$$
(s_1,a_1,\ldots,s_t,a_t),
$$

不能看到未来信息：

$$
(s_{t+1},a_{t+1},\ldots,s_T,a_T).
$$

---

# 三、为什么第一版建议先使用 GRU？

虽然 Transformer 与 Decision Transformer 的结构更加一致，但第一版实验建议优先使用 GRU，主要原因包括：

1. DTRD 本身采用双层优化，训练过程已经比较复杂；
2. reward model 的梯度需要通过 DT 的一次参数更新反向传播；
3. 如果直接使用较大的 Transformer，会显著增加显存和计算开销；
4. 序列长度较长时，Transformer 的复杂度也更高；
5. GRU 参数量更小，更适合验证“历史信息是否有效”这一核心假设。

因此可以先验证：

$$
\text{DTRD-MLP}
\quad\text{vs.}\quad
\text{DTRD-GRU}.
$$

如果 GRU 版本有效，再进一步尝试 causal Transformer。

---

# 四、损失函数不需要发生根本变化

这个方案最大的优势是：

> 只修改 reward model 的输入形式，其余 DTRD 的训练框架可以基本保持不变。

不需要额外引入多切分损失、区间一致性损失或多尺度损失。

## 1. 代理奖励预测

原始 DTRD：

$$
\hat r_t=f_\phi(s_t,a_t).
$$

修改后：

$$
\boxed{
\hat r_t=f_\phi(\tau_{1:t})
}
$$

## 2. Shaped Return-to-Go

仍然按照 DTRD 的方法，根据代理奖励重新计算 shaped RTG：

$$
\boxed{
\hat g_t
=
g_t
-
\sum_{i=1}^{t-1}\hat r_i
}
$$

其中：

- $g_t$ 是原始 return-to-go；
- $\hat r_i$ 是 reward model 生成的代理奖励；
- $\hat g_t$ 是重分配后的 return-to-go。

在完全延迟奖励任务中，原始 RTG 可能近似为：

$$
g_1=g_2=\cdots=g_T=G.
$$

随着已经完成的代理奖励被逐步扣除，$\hat g_t$ 会随时间发生变化，从而给 DT 提供更加细粒度的条件信息。

## 3. 回报守恒损失

DTRD 要求一条轨迹上所有代理奖励的总和接近真实轨迹回报。

原始形式为：

$$
\mathcal L_{\mathrm{redis}}
=
\sum_{\tau}
\left(
G(\tau)
-
\sum_{t=1}^{T}
f_\phi(s_t,a_t)
\right)^2.
$$

修改后变为：

$$
\boxed{
\mathcal L_{\mathrm{redis}}
=
\sum_{\tau}
\left(
G(\tau)
-
\sum_{t=1}^{T}
f_\phi(\tau_{1:t})
\right)^2
}
$$

它要求：

$$
\sum_{t=1}^{T}\hat r_t
\approx
G(\tau).
$$

也就是说，代理奖励可以重新分配到不同时间步，但不能随意改变整条轨迹的总回报。

## 4. 下层 DT 优化

使用 history-conditioned reward model 生成的 shaped RTG 训练 Decision Transformer。

下层优化可以写成：

$$
\theta^*(\phi)
=
\arg\min_\theta
\mathcal L_{\mathrm{train}}(\theta,\phi).
$$

对于连续动作，动作预测损失可以写为：

$$
\mathcal L_{\mathrm{train}}
=
\sum_t
\left\|
\pi_\theta
\left(
\hat g_t,
s_{\leq t},
a_{<t}
\right)
-
a_t
\right\|_2^2.
$$

对于离散动作，通常使用交叉熵损失：

$$
\mathcal L_{\mathrm{train}}
=
-
\sum_t
\log
\pi_\theta
\left(
a_t
\mid
\hat g_t,
s_{\leq t},
a_{<t}
\right).
$$

## 5. 上层 reward model 优化

保持 DTRD 原有的双层优化目标：

$$
\boxed{
\min_\phi
\left[
\mathcal L_{\mathrm{val}}
\left(
\theta^*(\phi),
\phi
\right)
+
\lambda
\mathcal L_{\mathrm{redis}}
\right]
}
$$

其中：

- $\mathcal L_{\mathrm{val}}$：验证当前 reward redistribution 是否能够帮助 DT 在验证集上更准确地预测动作；
- $\mathcal L_{\mathrm{redis}}$：保证代理奖励之和与轨迹真实回报一致；
- $\lambda$：控制回报守恒约束的强度。

因此，你的改进不需要新增损失函数。

核心变化只有：

$$
\boxed{
f_\phi(s_t,a_t)
\longrightarrow
f_\phi(\tau_{1:t})
}
$$

---

# 五、完整算法形式

可以暂时将该方法称为：

> History-Conditioned DTRD，简称 History-DTRD。

## 原始 DTRD

$$
\begin{aligned}
\hat r_t
&=
f_\phi(s_t,a_t),\\
\hat g_t
&=
g_t-\sum_{i<t}\hat r_i,\\
\theta'
&=
\theta
-
\alpha_\theta
\nabla_\theta
\mathcal L_{\mathrm{train}}(\theta,\phi),\\
\phi
&\leftarrow
\phi
-
\alpha_\phi
\nabla_\phi
\left[
\mathcal L_{\mathrm{val}}(\theta',\phi)
+
\lambda
\left(
G-\sum_t\hat r_t
\right)^2
\right].
\end{aligned}
$$

## History-DTRD

只替换代理奖励模型：

$$
\boxed{
\begin{aligned}
h_t
&=
\operatorname{GRU}_\phi
\left(
h_{t-1},
[s_t,a_t]
\right),\\
\hat r_t
&=
g_\phi(h_t).
\end{aligned}
}
$$

或者使用 causal Transformer：

$$
\boxed{
\begin{aligned}
h_t
&=
\operatorname{CausalTransformer}_\phi
(s_1,a_1,\ldots,s_t,a_t),\\
\hat r_t
&=
g_\phi(h_t).
\end{aligned}
}
$$

后续 shaped RTG 计算、DT 训练和双层优化过程保持不变。

---

# 六、必须使用 causal 序列模型

reward model 不能使用双向 Transformer。

如果使用双向模型，计算：

$$
\hat r_t
$$

时可能看到未来状态和未来动作：

$$
s_{t+1},a_{t+1},\ldots,s_T,a_T.
$$

这会产生未来信息泄漏。

训练时完整轨迹已知，模型可以利用未来信息；但测试时未来轨迹尚未发生，无法获得相同输入，会导致严重的 train-test mismatch。

正确的推理流程应该是：

```text
已知历史 τ_{1:t-1}
        ↓
根据 shaped RTG 和当前状态预测动作 a_t
        ↓
执行动作 a_t，并观察下一状态 s_{t+1}
        ↓
reward model 根据 τ_{1:t} 预测代理奖励 r̂_t
        ↓
更新下一时间步的 shaped RTG
```

因此，预测动作 $a_t$ 时，只能使用之前的代理奖励：

$$
\hat r_1,\hat r_2,\ldots,\hat r_{t-1}.
$$

不能提前使用：

$$
\hat r_t.
$$

---

# 七、与“预测最终回报再作差”的区别

另一个可能的方案是先预测最终回报：

$$
\hat G_t
=
F_\phi(\tau_{1:t}),
$$

再通过差分得到代理奖励：

$$
\hat r_t
=
\hat G_t-\hat G_{t-1}.
$$

这种方案更接近 RUDDER。

但如果当前目标是对 DTRD 进行干净的改进，第一版不建议同时改变以下内容：

1. reward model 的输入；
2. reward model 的输出含义；
3. 代理奖励的生成方式；
4. reward model 的优化目标。

否则，即使实验性能提升，也很难判断提升究竟来自哪一项修改。

因此第一版建议只修改：

$$
\boxed{
f_\phi(s_t,a_t)
\longrightarrow
f_\phi(\tau_{1:t})
}
$$

并保持：

$$
\hat r_t=f_\phi(\tau_{1:t})
$$

直接输出逐步代理奖励。

后续可以将“prefix return prediction + difference”作为单独的对照实验。

---

# 八、哪些任务最可能受益？

这个方法最可能在以下类型的任务中有效。

## 1. 部分可观测任务

当前观测不能完整表示真实环境状态，需要根据历史信息推断隐藏变量。

## 2. 状态混淆任务

不同任务阶段可能出现相同或非常相似的状态动作对。

例如智能体多次经过同一个位置，但第一次经过与完成子目标后再次经过的意义不同。

## 3. 具有前置子目标的任务

例如：

- 先拿钥匙，再开门；
- 先按开关，再通过通道；
- 先完成准备动作，再执行最终动作；
- 先访问某一区域，再到达目标。

在这类任务中，当前动作的价值依赖之前是否完成了某个子目标。

## 4. 存在循环或重复状态的任务

智能体可能多次经过相同状态：

$$
s_i\approx s_j,
$$

但由于历史不同，两次状态对应的任务阶段并不相同。

历史 reward model 能够区分：

$$
f_\phi(\tau_{1:i})
\neq
f_\phi(\tau_{1:j}).
$$

---

# 九、哪些情况下可能没有效果？

## 1. 当前状态已经是完整的 Markov 状态

如果 $s_t$ 已经包含：

- 是否拿到钥匙；
- 是否完成子目标；
- 当前任务阶段；
- 当前时间步；
- 所有必要的环境信息；

那么：

$$
f_\phi(s_t,a_t)
$$

理论上已经能够判断当前动作的贡献。

此时完整历史可能只是冗余输入，甚至会导致过拟合。

## 2. 离线数据量不足

序列模型的容量通常明显高于单步 MLP。

如果离线轨迹数量有限，可能出现：

- 训练集回报重建误差较低；
- 训练集动作预测性能较好；
- 测试环境中的真实性能下降。

## 3. 双层优化成本增加

DTRD 需要 reward model 的梯度通过 DT 的一次参数更新进行反向传播。

将简单的 MLP reward model 替换为 GRU 或 Transformer 后，会增加：

- 显存占用；
- 训练时间；
- 梯度传播长度；
- 双层优化的不稳定性；
- 长轨迹上的计算开销。

因此第一版模型应该尽量小。

## 4. Shaped RTG 可能变成信息侧信道

如果历史 reward model 的容量过强，它可能不再学习可解释的代理奖励，而是把历史轨迹信息编码到 shaped RTG 中。

例如，某个特殊数值并不代表真实的剩余回报，而只是向 DT 传递：

> 下一步应该向左走。

此时 shaped RTG 变成了 reward model 和 DT 之间的“暗号”。

因此需要检查：

- 代理奖励在不同随机种子下是否稳定；
- 代理奖励是否集中在合理的关键事件附近；
- 更换 DT 初始化后是否仍然有效；
- 训练好的 reward model 能否迁移到另一个 DT；
- 代理奖励是否具有基本的时间和任务语义。

---

# 十、建议的实验设置

不要只比较：

$$
\text{DTRD}
\quad\text{vs.}\quad
\text{History-DTRD}.
$$

建议至少设置以下版本。

| 方法 | Reward model 输入 |
|---|---|
| DTRD-Markov | $(s_t,a_t)$ |
| DTRD-Window | 最近 $K$ 步历史 |
| DTRD-GRU | 完整历史 $\tau_{1:t}$ |
| DTRD-Transformer | 完整历史 $\tau_{1:t}$ |

## 1. 固定窗口版本

定义：

$$
\hat r_t
=
f_\phi
\left(
s_{t-K+1:t},
a_{t-K+1:t}
\right).
$$

设置：

$$
K\in\{1,4,16,64,\text{full}\}.
$$

其中：

- $K=1$ 对应原始 DTRD；
- $K>1$ 表示使用局部历史；
- $\text{full}$ 表示使用完整历史。

可以分析：

$$
\text{上下文长度}
\longrightarrow
\text{最终策略性能}.
$$

如果性能随着上下文长度增加而提升，说明历史信息确实有价值。

## 2. 模型结构消融

比较：

- MLP；
- GRU；
- LSTM；
- causal Transformer。

这样可以区分：

> 提升来自历史信息，还是仅仅来自更大的模型容量。

## 3. 关键对照实验

建议至少包含：

1. 标准 DT；
2. 原始 DTRD；
3. DTRD-Window；
4. DTRD-GRU；
5. DTRD-Transformer；
6. RUDDER + DT；
7. 原始稠密奖励 DT，作为 oracle 上界。

---

# 十一、一个低成本的前期诊断实验

正式修改 DTRD 之前，可以先训练两个轨迹回报预测器。

## 单步预测器

$$
\hat G_t^{\mathrm{Markov}}
=
q(s_t,a_t).
$$

## 历史预测器

$$
\hat G_t^{\mathrm{History}}
=
q(\tau_{1:t}).
$$

然后比较验证集回报预测误差：

$$
\operatorname{MSE}_{\mathrm{Markov}}
$$

和：

$$
\operatorname{MSE}_{\mathrm{History}}.
$$

如果：

$$
\operatorname{MSE}_{\mathrm{History}}
\ll
\operatorname{MSE}_{\mathrm{Markov}},
$$

说明历史信息中包含当前 $(s_t,a_t)$ 无法表达的内容，该任务比较适合 History-DTRD。

如果：

$$
\operatorname{MSE}_{\mathrm{History}}
\approx
\operatorname{MSE}_{\mathrm{Markov}},
$$

说明当前状态已经基本能够表达任务进展，引入完整历史的收益可能有限。

需要注意，这个实验只能作为诊断，不能替代正式的 DTRD 训练结果。

---

# 十二、这个方向的创新性

## 1. 作为前期实验或课程项目

这个方向完全值得做。

它具有以下优点：

- 修改明确；
- 代码改动相对有限；
- 实验容易控制；
- 研究问题容易解释；
- 可以直接分析上下文长度的影响。

## 2. 作为论文中的一个方法模块

也是合理的。

尤其是如果能够证明：

$$
\text{任务的历史依赖性越强}
\quad\Longrightarrow\quad
\text{History-DTRD 的优势越明显},
$$

就可以形成较完整的实验结论。

## 3. 作为完整论文的唯一创新

目前可能稍显不足。

因为已有工作已经使用序列模型进行奖励重分配，例如 RUDDER；也已有工作研究非 Markov 奖励建模。

因此，完整论文不能只强调：

> 我们把 MLP 换成了 GRU 或 Transformer。

更有价值的研究问题包括：

1. DTRD 的 Markov reward assumption 在什么任务中会失效？
2. 如何衡量任务或数据集的 reward history dependence？
3. 历史信息在多长的上下文范围内最有效？
4. 如何避免 shaped RTG 变成信息侧信道？
5. 历史 reward model 是否能提高跨任务或跨数据集泛化？
6. 在完全可观测任务和部分可观测任务中，效果是否不同？

---

# 十三、推荐的研究路线

## 第一阶段：复现基线

首先稳定复现：

$$
\text{DT}
$$

和：

$$
\text{DTRD}.
$$

确认原始结果和训练流程基本正确。

## 第二阶段：只修改 reward model 输入

将：

$$
f_\phi(s_t,a_t)
$$

修改为：

$$
f_\phi(\tau_{1:t}).
$$

第一版优先使用小型 GRU。

其余内容全部保持不变：

- shaped RTG 的计算；
- 回报守恒损失；
- DT 动作预测损失；
- 验证集上层目标；
- 单步梯度近似；
- 数据划分和训练超参数。

## 第三阶段：上下文长度消融

设置：

$$
K\in\{1,4,16,64,\text{full}\}.
$$

分析历史长度是否影响性能。

## 第四阶段：模型结构消融

比较：

$$
\text{GRU}
\quad\text{vs.}\quad
\text{LSTM}
\quad\text{vs.}\quad
\text{causal Transformer}.
$$

## 第五阶段：分析历史信息何时有效

将任务划分为：

- 状态信息充分的 Markov 任务；
- 存在状态混淆的任务；
- 部分可观测任务；
- 具有前置子目标的任务；
- 长时延稀疏奖励任务。

验证 History-DTRD 的优势是否与任务历史依赖程度相关。

---

# 十四、最终建议

第一版方法应当尽量简单：

$$
\boxed{
\begin{aligned}
\text{原始 DTRD：}\quad
&\hat r_t=f_\phi(s_t,a_t),\\[4pt]
\text{History-DTRD：}\quad
&h_t
=
\operatorname{GRU}_\phi
\left(
h_{t-1},
[s_t,a_t]
\right),\\
&\hat r_t
=
g_\phi(h_t).
\end{aligned}
}
$$

其余 DTRD 组件全部保持不变：

- shaped RTG；
- reward conservation loss；
- DT action loss；
- validation-based upper-level objective；
- bilevel optimization。

这样能够干净地回答：

$$
\boxed{
\text{在 DTRD 中，为 reward model 引入历史条件是否有效？}
}
$$

这个研究问题相比复杂的多尺度、多网络、多损失方案更加可靠，也更容易通过实验得到清晰结论。
