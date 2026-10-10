# FLow-GRPO

## 1. Flow-GRPO 一句话理解

**Flow-GRPO 是把 flow matching 视频生成模型的去噪采样过程改造成一个可强化学习优化的随机策略，然后用 GRPO 根据生成结果的奖励来更新模型。**

在 World-R 1 里，它的作用是：

```text
视频模型生成多个候选视频
→ 用 3D 一致性奖励和视觉质量奖励打分
→ Flow-GRPO 根据分数更新视频模型
→ 让模型更倾向生成 3D 一致的视频
```

---

## 2. 为什么普通 flow matching 不能直接做 RL？

普通 flow matching 模型的采样过程通常是一个**确定性 ODE**：

[  
dx_t = v_t dt  
]

这里的 (v_t) 是模型预测的 velocity field，也就是“当前 latent 应该往哪个方向移动”。

可以理解为：

```text
当前噪声状态 x_t
→ 模型预测一个方向 v_t
→ 沿着这个方向一步步走
→ 最后得到生成结果 x_0
```

问题是：**强化学习需要随机性**。

因为 RL 需要做两件事：

```text
1. exploration：探索不同生成结果
2. advantage estimation：比较哪些生成结果更好
```

如果采样是完全确定的，那么同一个 prompt、同一个初始条件下，模型总是走同一条路径。这样就很难说：

```text
这次生成比那次生成好多少？
哪些采样动作应该被强化？
哪些采样动作应该被削弱？
```

所以 Flow-GRPO 的第一步，就是把原本确定性的 flow sampling 变成随机采样。

---

## 3. 核心改造一：从 ODE 变成 SDE

原来的确定性 ODE 是：

[  
dx_t = v_t dt  
]

Flow-GRPO 把它改成随机微分方程 SDE：

[  
dx_t =  
\left[  
v_t(x_t)  
+  
\frac{\sigma_t^2}{2 t}  
\left(  
x_t + (1-t)v_t(x_t)  
\right)  
\right]dt  
+  
\sigma_t dw  
]

这条公式可以拆成三部分理解。

---

### 3.1 第一部分：原来的生成方向

[  
v_t(x_t)  
]

这是 flow matching 模型原本预测的方向。

它告诉模型：

```text
当前 latent x_t 应该往哪里走
才能从噪声逐渐变成清晰视频
```

---

### 3.2 第二部分：补偿项

[  
\frac{\sigma_t^2}{2 t}  
\left(  
x_t + (1-t)v_t(x_t)  
\right)  
]

这一项看起来复杂，但直觉上可以理解为：

> **既然我们给采样过程加了随机噪声，就需要额外加一个校正项，避免采样分布偏离原来预训练 flow 模型的分布。**

也就是说，Flow-GRPO 不是随便往采样过程中加噪声。它希望做到：

```text
加入随机性
但不要破坏原模型原本学到的生成分布
```

所以这一项的作用是**保持 marginal distribution 尽量不变**。

---

### 3.3 第三部分：随机噪声项

[  
\sigma_t dw  
]

这里：

```text
σ_t：控制噪声强度
dw：Wiener process，也就是连续时间里的随机噪声
```

这部分负责给采样过程引入随机性。

直观理解：

```text
原来：每一步只能沿固定路线走
现在：每一步在主要方向附近带一点随机扰动
```

这样模型就能对同一个 prompt 生成一组不同的视频样本，为后续 GRPO 比较好坏提供基础。

---

## 4. 离散化后：采样过程变成 policy

论文把上面的 SDE 离散化，得到：

# [  
x_{t+\Delta t}

x_t  
+  
\left[  
v_\theta(x_t,t)  
+  
\frac{\sigma_t^2}{2 t}  
\left(  
x_t + (1-t)v_\theta(x_t,t)  
\right)  
\right]\Delta t  
+  
\sigma_t\sqrt{\Delta t}\epsilon  
]

其中：

[  
\epsilon \sim \mathcal{N}(0,I)  
]

这条公式的意思是：

```text
下一步 latent
=
当前 latent
+
模型预测的主要更新方向
+
分布保持用的校正项
+
随机高斯噪声
```

这一步非常关键，因为它把生成模型变成了强化学习里的 policy：

[  
\pi_\theta  
]

为什么？

因为现在每一步从 (x_t) 到 (x_{t+\Delta t}) 不是完全确定的，而是一个带概率的转移。也就是说，模型在每个 denoising step 都是在“采样一个动作”。

可以对应成 MDP：

|RL 概念|在 Flow-GRPO 里的含义|
|---|---|
|state|当前 latent (x_t) 和条件 (c)|
|action|生成下一步 latent (x_{t+\Delta t})|
|policy|视频生成模型的随机采样分布 (\pi_\theta)|
|trajectory|完整去噪链 (x_T \rightarrow x_0)|
|reward|最终视频的打分 (R(x_0,c))|

---

## 5. 核心改造二：用 GRPO 做组内比较

Flow-GRPO 不像 PPO 那样额外训练一个 critic/value model，而是使用 GRPO 的组内归一化思想。

对于同一个条件 (c)，也就是同一个 prompt，采样 (G) 个生成轨迹：

[  
{x^i}_{i=1}^{G}  
]

每个轨迹最终都会生成一个结果 (x_0^i)，然后用 reward function 打分：

[  
R(x_0^i,c)  
]

在 World-R 1 里，这个 reward 包括：

```text
3D 一致性奖励
+
相机轨迹奖励
+
meta-view 结构评价
+
视觉质量奖励
```

然后对同一组里的 reward 做归一化，得到 advantage：

[  
\hat{A}_t^i =  
\frac{  
R(x_0^i,c) - mean({R(x_0^i,c)}_{i=1}^{G})  
}{  
std({R(x_0^i,c)}_{i=1}^{G})  
}  
]

这条公式的意思很简单：

```text
如果某个视频比同组平均水平好
→ advantage 为正
→ 增加这条生成轨迹的概率

如果某个视频比同组平均水平差
→ advantage 为负
→ 降低这条生成轨迹的概率
```

例如同一个 prompt 采样 8 个视频：

|视频|Reward|相对水平|更新方向|
|---|--:|---|---|
|video 1|0.80|高于平均|强化|
|video 2|0.75|高于平均|强化|
|video 3|0.50|接近平均|小幅更新|
|video 4|0.20|低于平均|抑制|

所以 GRPO 的核心思想是：

> **不需要知道绝对分数有多好，只需要知道同一组样本里谁更好。**

---

## 6. 核心改造三：用 clipped objective 稳定更新

论文中的优化目标是：

# [  
J(\theta)

## \mathbb{E}_{c,{x^i}}  
\left[  
\frac{1}{T}  
\sum_{t=0}^{T-1}  
\left(  
L_{clip}(r_t^i,\hat{A}_t^i)

\beta D_{KL}(\pi_\theta || \pi_{ref})  
\right)  
\right]  
]

这里有两个关键部分。

---

### 6.1 (L_{clip})：防止策略更新太猛

[  
L_{clip}(r_t^i,\hat{A}_t^i)  
]

这里的 (r_t^i) 是 probability ratio：

```text
当前策略生成这一步的概率
/
旧策略生成这一步的概率
```

如果某条轨迹 reward 很高，模型当然应该提高它的概率。

但不能提高得太猛，否则模型会崩：

```text
过度追求 reward
→ 生成分布偏移
→ 视频质量下降
→ reward hacking
```

所以 PPO-style clipping 的作用是限制单次更新幅度。

直观理解：

```text
好的样本可以强化
但不能一下子强化过头
```

---

### 6.2 KL 约束：防止偏离原始视频模型

[  
D_{KL}(\pi_\theta || \pi_{ref})  
]

这里的 (\pi_{ref}) 是 reference policy，通常就是原始预训练模型。

KL 项的作用是：

```text
让新模型不要离原始模型太远
```

这在视频生成中特别重要。因为原始视频模型已经有很强的视觉生成能力，如果 RL 只追求某个 reward，可能会牺牲画质、多样性和自然运动。

所以 KL 约束相当于在说：

```text
你可以变得更 3D 一致
但不要忘记原来怎么生成高质量视频
```

(\beta) 控制这个约束的强度。

---

## 7. Flow-GRPO 在 World-R 1 中的完整流程

结合 World-R 1，可以把 Flow-GRPO 的训练过程理解成下面这样：

```text
Step 1：输入一个文本 prompt c

Step 2：用当前视频模型采样 G 个候选视频

Step 3：对每个视频计算 reward
        包括 3D 一致性、轨迹对齐、视觉质量等

Step 4：同组内比较 reward
        得到每个视频的 advantage

Step 5：如果某个视频比同组更好
        提高对应去噪轨迹的概率

Step 6：如果某个视频比同组更差
        降低对应去噪轨迹的概率

Step 7：用 clipping 和 KL 约束防止模型更新过猛

Step 8：重复训练
        模型逐渐更倾向生成 3D 一致且高质量的视频
```

---

## 8. 为什么 World-R 1 适合用 Flow-GRPO？

因为 World-R 1 的目标不是监督学习式的：

```text
给定文本 → 拟合某个标准答案视频
```

而是偏向偏好优化：

```text
给定文本
→ 生成多个可能视频
→ 哪个视频更符合 3D 一致性和视觉质量
→ 强化那个更好的生成路径
```

这正好符合 RL 的范式。

尤其是 World-R 1 的 reward 很难写成普通监督 loss。比如：

```text
这个视频换视角后是否结构合理？
这个 3DGS 是否稳定？
相机轨迹是否符合 prompt？
视频是否美观？
```

这些都更像是“评价生成结果好不好”，而不是“逐像素告诉模型标准答案是什么”。

所以 Flow-GRPO 的价值在于：

> **它允许作者用外部评价器给生成视频打分，然后直接用这些分数优化视频生成模型。**

---

## 9. 和普通监督微调的区别

|对比项|普通监督微调|Flow-GRPO|
|---|---|---|
|训练信号|标准答案视频|reward 打分|
|是否需要 GT 视频|通常需要|不一定需要|
|优化目标|模仿数据|生成更高 reward 的结果|
|适合任务|有明确标注的数据拟合|偏好优化、质量优化、几何约束优化|
|在 World-R 1 中的作用|不适合直接表达 3D 一致性|可以用 3 D/VLM reward 引导模型|

---

## 10. 最核心的理解

Flow-GRPO 可以压缩成三句话：

```text
1. 把确定性的 flow 采样改成带噪声的随机采样。
2. 对同一个 prompt 生成一组视频，用 reward 比较谁更好。
3. 强化高分视频的生成轨迹，同时用 KL 防止模型偏离原始能力。
```

放到 World-R 1 中，就是：

> **Flow-GRPO 负责把“3 D 一致性奖励”真正转化成视频模型参数的更新。**