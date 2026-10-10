# TokLIP

## Motivation

现有的方法试图讲 CLIP 的高层连续特征离散化，导致语义的损失比较严重，应该 _将 VQ token 语义化_

优势：
- 把理解和生成的训练目标拆开了
- 把 CLIP 级别语义引入了统一自回归模型
- 可以直接复用最强的 VQ tokenizer，而不必在语义量化上折腾

![[assets/Pasted image 20260327125348.png]]

作者发现，如果直接从零开始训练 causal ViT，性能会明显掉。于是 token encoder 不是随机初始化，而是从 **预训练的双向 CLIP vision encoder** 初始化