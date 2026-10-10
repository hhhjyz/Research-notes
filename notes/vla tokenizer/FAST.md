# FAST：Frequency-space Action Sequence Tokenization

## **motivation：**
- Per-dimension, per-timestep binning scheme 的问题：学习 dexterous skills with high-frequency control 时，表现不佳
- action token 之间过于相似，影响 next toekn prediction objective 的有效性
- Action signal 需要在训练前被压缩，来降低连续 token 之间的相关性
    

## **method**

Based on DCT encoding(离散余弦变换)

压缩 highly redundant action signal $\rightarrow$ small number of high-information tokens

### Time-Series Compression via Discrete Cosine Transform

**DCT** :频域变换，它将连续信号表示为不同频率余弦元素的和。低频捕获信号的总体形状，而高频成分则反映急剧的跳跃。

### The FAST Tokenization Algorithm

1. 归一化：对每个独立动作维度的数值分布进行统计分析，提取其第 1 百分位数和第 99 百分位数，并将这两个锚点严格线性映射至 [-1, 1] 的区间、
```python
# 1. 沿时间维度（axis=1）进行 DCT 变换
dct_coeff = dct(action_chunk, axis=1, norm="ortho")
# 2. 缩放并四舍五入实现量化
dct_coeff = np.around(dct_coeff * self.scale)
```
1. 对每个动作维度做 DCT 变换，简单地省略不重要的系数，通过缩放和舍入操作来实现
2. 在舍入运算之后，DCT 系数矩阵通常是稀疏的，大多数项为零，每个动作维度仅保留少数几个显著的系数
3. Column-first flatten， 把所有动作维度的最低频分量排在最前面。由于自回归模型是按顺序预测下一个 Token 的，让模型优先预测决定动作整体形状的低频分量，能够使得模型在生成动作序列时更加稳定
```python
# __call__ 中的实现
elem.flatten()

# fit 中的实现
dct_tokens = [dct(a, axis=0, norm="ortho").flatten() for a in action_data]
```
5. BPE， 无损压缩
```python
# 1. 将数字转换为字符序列 (桥接步骤)
token_str = "".join(map(chr, np.maximum(elem.flatten() - self.min_token, 0).astype(int)))
# 2. 调用训练好的 BPE 模型进行分词压缩
tokens.append(self.bpe_tokenizer(token_str)["input_ids"])
```

![[Pasted image 20260325204928.png]]
