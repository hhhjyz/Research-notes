# EVATok

传统的 video tokenizer 在不同视频的时间块上应用统一的 token 分配，这常常在简单、静态或重复的片段上浪费 token，而在动态或复杂的片段上分配不足

以往的自适应方法（如基于启发式阈值搜索或批次内线性规划）往往忽略了全局的质量-成本平衡，或者将单样本的分配决策与固定批次预算强行绑定 ()()()。此外，现有的代理分词器方案存在训练与推理时的 gap（训练时遍历所有分配，推理时只用少部分），导致性能下降

现有的 video adaptive tokenizers：

- Heuristic threshold-based searches：不一定真的在全局上实现最好的“质量—成本”平衡
    
- Mini-batch ILP：在一个 batch 内做整数规划，让平均预算满足约束。但是没有解决对于每个样本，确定适应样本固有复杂性的最佳分配，实现最优的 adaptive tokenization
    

**如何定义 optimal assignment identification problem？**

表述为一个 tractable maximum proxy reward assignment identification task, proxy reward 用于衡量 reconstruction quality and cost(token length)

### Method

![](https://dcnghgzhlm4y.feishu.cn/space/api/box/stream/download/asynccode/?code=NGFiMTc0NjBiYmMyOWEwMzQ2NjdhOTJkMjI5NjA1MjhfSGRrVFdJd3dqZHJkdWtKcXRMaXl2elN1WG9NbnpSTkpfVG9rZW46VkFGaWJiNmk1b0VoaFZ4Nnh3ZGM4T0ZOblhiXzE3NzQ0MTg3Mjk6MTc3NDQyMjMyOV9WNA)

流程：

- 第一步：训练一个 proxy tokenizer：这个 tokenizer 能在不同 token assignment 下重建视频。
    
- 第二步：枚举候选 assignment，找到每个视频的最优 assignment。对于每个视频，遍历所有可能的分配方式，计算 proxy reward，挑出 reward 最高的那个，作为该视频的“最优 assignment”。
    
- 第三步：训练一个轻量 router：因为真实使用时不可能每来一个视频都暴力搜索所有 assignment，所以作者训练一个小模型 router，让它直接根据输入视频预测最优 assignment。
    
- 第四步：再训练最终的 adaptive tokenizer。最终 tokenizer 不再用固定长度，而是根据 router 预测出来的 assignment，对视频进行自适应编码。
    

使用 Q-Former 风格的 1 D tokenizer 来构建 adaptive video tokenizer

#### Problem Setup

在时间上，下采样四倍；将视频切分成 T casual blocks。对每个时间块 t，都要从候选集合 K 里选一个 token 数 $$k_$$ ,整个视频的一个 token 分配方案就写成 $$a = (k_1, \dots, k_T$$ ,总 token 长度 $$L(a) = \sum_{t=1}^T k_$$

1. #### Training a Proxy tokenizer
    

这个 proxy tokenizer 可以根据 randomly sampled assignment a 来重建视频，

**Model architecture：**

![](https://dcnghgzhlm4y.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTYyZGU4ZTI5N2VmMzA2MjAzNTE1Mjg1NzdjMzgwY2RfQ2ZSeHAyWWlHeEpwcFkwQUg1dVNvTEY0d2E4Z0l4V2NfVG9rZW46RXE3eWJnR1cyb2p1MjB4Z2lLS2NnV0VsblJiXzE3NzQ0MTg3Mjk6MTc3NDQyMjMyOV9WNA)

采用了基于 Q-Former 风格的 1D tokenizer 架构

- Patchify：将输入的视频帧在空间和时间维度上进行 patchif），通过一个简单的线性层将其转化为 3D embeddings
    
- 假设此时随机采样到的分配方案为 $$a=(q_1,q_2,q_3,q_4$$ （表示四个时间块分别需要 $$q_1, q_2, q_3, q_4$$ 个 token）。模型会根据该方案，从对应时间块的 3 D 特征进行 2 D 池化，以此初始化对应数量的 1D Queries
    
- 这些 1D queries 通过 Q-Former 编码器，提取 3D embeddings 特征中的视觉信息，并经过 VQ 操作，转化为离散的 1D token 。这里使用了 Temporally causal attention masks，确保每个时间块的 token 只能“看到”当前及过去的特征，无法编码未来时间块的信息
    
- As for the decoder, 模型使用每个时间块的第一个 1D token 来初始化 3D queries。随后经过类似的因果解码过程，输出最终的 3D 特征，并通过 linear project 和 reshape 操作还原为视频帧
    

不再使用 tail-token-dropping strategy

**Training recipe：**

- $$\mathcal{L}_{vqgan}$$：包含 L 1 重建损失、感知损失、GAN Loss（patch gan）以及 VQ codebook loss。
    
- $$\mathcal{L}_{\text{align}} = -\frac{1}{N} \sum_{n=1}^{N} \text{sim}(f_n^{\text{dec},l}, \phi(f_n^{\text{sem}})) $$：前文提到的特征对齐损失 ()。
    
- $$\mathcal{L}_{entropy}$$：熵损失，用于优化 codebook 的使用率
    

2. #### Dataset Curation for Router Training
    

穷举的成本很大，所以训练一个轻量化的 router 来预测 optimal assignment

$$R_{\text{proxy}} = w_q Q(\mathcal{E}_{\text{proxy}}, x, a) - w_l L(a)$$： 在重建质量和 token 长度成本之间寻找最佳平衡

- $$Q(\mathcal{E}_{\text{proxy}}, x, a$$：代表使用 proxy tokenizer $$\mathcal{E}_{\text{proxy}}$$，在分配方案 $$$$ 下，对视频 $$$$ 的重建质量。质量越高，奖励越大。
    
- $$L(a$$：代表分配方案 a$消耗的 token 总长度。长度越长，惩罚越大（注意前面是减号）。
    
- $$w_$$ **和** $$w_$$ ：weight
    

最优分配方案：$$a^* = \text{argmax}_{a \in A} R_{\text{proxy}}$$

质量 $$$$ 具体使用的是归一化后的 LPIPS,成本 $$$$ 使用的是归一化后的长度,必须归一化是因为 LPIPS 的数值通常在 0.1 左右，而长度往往是几百上千，不在同一量级，无法直接用权重相加减。

对于每个视频，只保留 proxy reward 最大的那个方案

3. #### Router Training
    

采用类似 ViT 的架构, 输入视频首先被 patchifies into 3D visual embeddings,在特征序列前面加上一个特殊的分类标识符 `[CLS]` embedding,经过网络处理后，直接从 `[CLS]` 提取特征，输出该视频属于每一种候选分配方案的概率

使用经典的 cross-entropy loss 进行训练

4. #### Adaptive Length Video Tokenizer
    

每次输入一个视频，直接让刚刚训练好的 Router 预测出它的最优分配。然后，adaptive tokenizer 只针对这唯一的一个分配方案进行重构学习，消除了训练和推理的不一致性。

引入了 Video Semantic Discriminator，借用了一个冻结的预训练 VideoMAE-B 模型来处理输入视频。提取它的多层特征后，送入可训练的 1D CNN 头部，用来输出真假对立的逻辑值（fake/real logits），以此进行对抗训练