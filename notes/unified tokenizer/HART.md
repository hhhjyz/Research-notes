# HART: Hybrid Autoregressive Transformer

现有 AR 模型主要受限于两个问题： 
1. 离散 tokenizer 的图像重建质量较差：codebook 容量有限
2. 直接生成高分辨率的图像训练代价过高：注意力成本很高

图像的整体结果和细节适用两种不同的机制来建模

![[Pasted image 20260402201804.png]]

## tokenizer 训练

**continuous latent = summed discrete latent + residual latent**

1. input x 经过 CNN-based visual decoder 得到 continuous latent
2. continuous tokens 做 multi-scale VQ, resize 并 sum up
3. 作差得到 residual tokens
4. _交替训练：_
	- 从预训练好的 VAR tokenizer 初始化 encoder、quantizer、decoder，**冻结 encoder 和 quantizer**，只训练 decoder
	- 要么不经过 VQ quantizer，直接把 continuous tokens 给 decoder。要么先经过 quantizer，得到 discrete tokens， 重建

![[Pasted image 20260402220653.png]]

1. text prompt 经过一个 LLM text encoder 编码成 text tokens, 训练时，文本 token 和视觉 token 拼接起来送入 AR transformer， 得到 discrete token 加上 residual diffusion 需要的最后一层 hidden states
2. multi-scale discrete tokens 累加求和
3. 引入一个很小的 diffusion MLP 来根据 AR transformer 最后一层的 hidden states 和最后一个 VAR sampler step 预测出的 discrete tokens 来建模 residual tokens（_学习在噪声条件下恢复 residual tokens_ ）
4. 相加得到 continuous tokens
5. 送入 decoder 解码

## inference

- 输入文本 prompt
- 文本编码成 text tokens
- AR transformer 分 stage 生成 discrete visual tokens
- 得到粗略但结构正确的 continuous 主干
- residual diffusion 用少量步数补齐细节
- decoder 输出最终高分辨率图像

