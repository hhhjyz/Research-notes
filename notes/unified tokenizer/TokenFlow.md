# TokenFlow

理解和生成需要不同粒度的 visual infomation
dual-codebook architecture

## Motivation

传统 VQ tokenizer 擅长 pixel-level image reconstruction，但是如果把它当作 feature extractor， 表现不如连续的 semantic feature. 因为这些 tokenizer 的 pre-training objective 是更好的低层重建质量，feature 主要编码的是 low-level information

从 CLIP 蒸馏离散 token，确实能更适合理解，但在生成上的表现又变差了

- VQKD： semantic grouping
- VQGAN： visual grouping

两个需求：
- high-level semantic understanding
- low-level visual reconstruction

## Method

将既有 high-level 又有 low-level 的 patch 映射到相同的 codebook index

- **输入**：给定一张原始图像。
- **特征提取（双编码器）**：图像同时输入到两个独立的编码器中。语义编码器提取高级语义特征 $\hat{z}_{sem}$；像素编码器提取底层细节特征 $\hat{z}_{pix}$ 。
- **联合量化（共享映射）**：系统维护两个独立的码本（语义码本和像素码本），但它们共享同一套索引 。对于提取出的特征，分别计算它们与各自码本中嵌入向量的 $L_2$ 距离 。
- **索引决策**：将语义距离和像素距离加权求和，找出使总距离最小的联合索引 $i^*$ 。
- **特征解码与输出**：利用确定的索引 $i^*$，分别从两个码本中取出对应的量化特征。语义解码器尝试将量化后的语义特征还原为教师模型（如 CLIP）提取的特征；像素解码器则将量化的像素特征重建为原始图像 。
- **下游应用**：在推理或下游任务训练时，这两个量化特征被拼接在一起，为多模态理解或生成提供统一表示


两个 Decoder，分别用于重建语义特征和原始图像
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{sem}} + \mathcal{L}_{\text{VQ}} + \mathcal{L}_{\text{pix}}$$
$$\mathcal{L}_{\text{pix}} = \ell_2(x, \hat{x}) + \mathcal{L}_{\text{P}}(x, \hat{x}) + \lambda_G \mathcal{L}_G(\hat{x})$$
其中 $\hat{x}$ 是像素解码器生成的图像，$x$ 是原图。这三项分别是：

- **$\ell_2(x, \hat{x})$ (Pixel-wise loss)**：最基础的像素级均方误差，保证整体颜色和轮廓的准确性。
    
- **$\mathcal{L}_{\text{P}}(x, \hat{x})$ (Perceptual loss)**：感知损失（通常使用 LPIPS）。它在特征空间比较图像差异，能避免单纯使用 $\ell_2$ 导致的图像模糊问题，提升人类视觉感知上的相似度。
    
- **$\mathcal{L}_G(\hat{x})$ (Adversarial loss)**：对抗损失（GAN Loss），权重为 $\lambda_G$。通过判别器的对抗训练，逼迫解码器生成极其逼真的高频细节（如纹理、边缘）。

引入了一个Teacher Model来提取目标特征，$\mathcal{L}_{sem}$ 就是和 teacher feature 之间的 $L_2$ distance
$$z = \text{sg}[z - \hat{z}] + \hat{z}$$
$$\mathcal{L}_{\text{VQ}} = ||\text{sg}[\hat{z}] - z||_2^2 + \beta||\hat{z} - \text{sg}[z]||_2^2$$


### Visual Generation with TokenFlow

与传统的逐个 Token 预测（Next-token prediction）不同，这里采用的是 **Next-scale prediction**范式。

**Training Strategy**
- 架构建立在预训练的大语言模型之上。文本端使用模型原生的 BPE 分词器；视觉端使用 TokenFlow 提取图像 Token，经过一个 MLP 层映射后，与文本 Token 拼接送入模型。
    
- 由于是自回归生成，模型只在图像 Token 上计算交叉熵损失（Cross-entropy loss），而不干预文本部分的学习。
    
- **无分类器引导 (CFG) 支持**：为了在推理时启用 CFG（Classifier-free guidance），训练阶段引入了 10% 的条件丢弃率（将文本条件随机替换为空字符串，$p_{drop} = 0.1$）。
    
- 针对大规模 Transformer 训练容易出现的Loss spikes问题，引入了 QK-normalization和 norm re-ordering技术。
    

**Inference Strategy**

- 如果直接将传统的 top-k / top-p 采样策略搬到 Next-scale 范式中，往往会导致图像结构崩溃和局部图案重复。这是因为交叉熵损失主要关注 Top-1 预测，如果在同一个尺度内对每个 Token 进行独立的 top-k 采样，Token 之间会缺乏直接的关联性。
    
- **解决方案：Multi-step sampling**：
    1. **初始采样**：在一个尺度内，先使用较大的参数组合（$k_1$ 和 $p_1$）进行常规采样，保证生成的多样性。
    2. **Refinement**：将第一步采样的输出作为输入，在同一个尺度下，使用收紧的参数（$k_2 < k_1$ 且 $p_2 < p_1$）进行第二轮采样。
    

---

### Multimodal Understanding with TokenFlow

这一节阐述了如何将 TokenFlow 作为纯视觉编码器，接入到现有的多模态大语言模型中。

- **架构对齐**：TokenFlow 在这里充当多尺度 VQ 分词器。量化后的特征被直接喂给预训练的 LLM，整体训练对齐了经典的 **LLaVA-1.5** 范式。
    
- **特征输入维度的消融实验**：TokenFlow 本质上会提取多个尺度的特征。作者在此验证了三种特征喂入策略：
    
    1. 输入所有尺度的特征 (Feature from all scales)
    2. **仅输入最终尺度的特征 (Final-scale feature only)**
    3. 输入所有尺度的残差特征 (Residual features from all scales)
        
- **核心结论**：实验证明，**仅使用最终尺度特征**取得了最佳的整体性能。作者的直觉解释是：多模态理解任务高度依赖**高级语义信息**（这通常浓缩在网络的最终尺度中），而强制引入早期尺度的细粒度/残差特征，反而会引入不必要的视觉噪声，拖累模型的逻辑推理表现。