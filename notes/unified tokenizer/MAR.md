# MAR

应用自回归模型到连续值的空间，对自回归模型来说，建模一个满足要求的分布是最重要的

**关键创新**： 
- 不再使用 cross-entropy loss，定义一个 diffusion loss function 来建模每个 token 的概率

概率分布需要满足的两个条件：
- loss function 可以度量预测分布和真实分布之间的差别
- sampler 可以根据分布来采样

## Diffusion Loss

### Loss Function

![[assets/Pasted image 20260409171340.png]]
### Sampler

![[assets/Pasted image 20260409172300.png]]

每一步自回归生成 条件向量 z， 以 z 和 t 作为条件，对纯噪声进行去噪

## MAR

**random order**： 给 decoder layer 增加 position embedding，能够识别出预测的是哪个位置
**bidirectional attention**: 在中间层替换位置 token 的内容，对已知的 token 编码时加上位置编码

![[assets/Pasted image 20260409184234.png]]