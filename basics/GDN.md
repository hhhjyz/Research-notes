# Hybrid LLM 之 Gated DeltaNet

## Attention 计算

$$
Attn(Q,K,V)=softmax(\frac{QK^T}{\sqrt d})V 
$$
$$
MultiHead(Q,K,V)=Concat(head_1,\cdots,head_H)W^O
$$
## Linear Attention

![[GDN-1792056723644.webp]]
去掉 softmax 不能立刻降低计算复杂度， 但是输入和输出之间是矩阵相乘的形式，可以进行代数操作

**推理时**：
$$
o_t=\sum_{j=1}^t v_j(k_j^Tq_t)=\sum_{j=1}^t(v_jk_j^T)q_t
$$
定义状态矩阵 $S_t=\sum^t_{j=1}v_jk^T_j$ ,可以进一步写成 $S_t=S_{t-1}+v_tk_t^T, o_t=S_tq_t$
可以看到线性 attention 本质上是一个具有矩阵值状态的线性 RNN
那么我们只需存储和更新 $S_t$，显著提高了效率：
- 时间复杂度从 $O(L^2d)$ 降低到 $O(Ld^2)$
- 空间复杂度从 $O(Ld)$ 降低到 $O(d^2)$

相当于使用 v 与 k 之间的外积的总和来存储历史信息
$$
M = \sum_{j=1}^t v_j k_j^T (M\in \mathbb R^{d\times d})
$$

## DeltaNet

根据期望值与实际结果之间的差异来调整模型的参数

不再是简单地累计键值外积，而是基于预测误差来更新自身状态

![[GDN-1792068122242.webp]]
- $\beta_t\in \mathbb R$ : 学习率
- $k_t\in \mathbb R^d$ : 输入
- $v_t \in \mathbb R_d$ : 目标
- $S_{t-1}k_t \in \mathbb R^d$ : 当前预测值
把 `S_{t-1} kt` 想象成从记忆中检索与当前 kt 相关的 “旧的 V”。当遇到与同一 K 关联的新值 Vt 时，不是直接盲目覆盖，而是谨慎地进行更新

**DeltaNet** 在每一步都最大限度地减少了 MSE，而传统 Linear Attention 采取的是线性损失函数

学习目标是让模型输出 $Sk_t$ 尽可能与真实值 $v_t$ 相似

![[GDN-1792069630080.webp]]
## Gated DeltNet

![[GDN-1792069661915.webp]]