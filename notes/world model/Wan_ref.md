---
tags:
status: active
arxiv:
github:
撰写时间: 2026-06-04
---
你现在读 [image2video_fast.py](/home/hhhjyz/research/lingbot-world/wan/image2video_fast.py) 时，最好不要把它当成一个全新的 pipeline，而是当成：

**“把 base 版 `image2video.py` 改造成可实时、可分块、可缓存的因果推理版本”**

所以最有效的读法是：只盯住“它和 base 不一样的地方”。

**先给你一个总判断**

`image2video_fast.py` 相比 `image2video.py`，主要不是改“控制逻辑”，而是改“推理机制”。

base 版重点是：
- 完整扩散采样
- 双模型切换
- 全序列一次性处理

fast 版重点是：
- 单模型因果推理
- chunk-by-chunk 生成
- KV cache
- prompt cache
- prewarm
- attention 窗口控制

所以你阅读时要始终问自己一句：

**“这一段是在改控制条件，还是在改推理效率？”**

大多数新增内容都属于后者。

---

## 你最该关注的差异点

### 1. 类和模型结构不同

base 版：
- [WanI2V](/home/hhhjyz/research/lingbot-world/wan/image2video.py:42)

fast 版：
- [WanI2VFast](/home/hhhjyz/research/lingbot-world/wan/image2video_fast.py:34)

最核心区别：

- base 用两个模型：
  - `low_noise_model`
  - `high_noise_model`
  位置在 [image2video.py](/home/hhhjyz/research/lingbot-world/wan/image2video.py:100)

- fast 只用一个：
  - `self.model = WanModelFast.from_pretrained(...)`
  位置在 [image2video_fast.py](/home/hhhjyz/research/lingbot-world/wan/image2video_fast.py:106)

这意味着你读 fast 版时，不需要再关注“高噪声/低噪声模型切换”那条线了。

---

### 2. 新增了缓存体系

这是 fast 版最值得重点读的地方。

#### T 5 cache
位置：
- [image2video_fast.py](/home/hhhjyz/research/lingbot-world/wan/image2video_fast.py:133)


作用：
- 相同 prompt 不重复跑 T 5 编码
- 直接复用已有文本特征

#### self-attention KV cache
在后面的 chunk 推理里使用，用来缓存历史视频 token 的 key/value。

#### cross-attention cache
也是缓存，但针对文本 cross-attention，避免每个 chunk 重算文本 K/V。

结论：
**fast 版把很多“重复算”的东西改成了“算一次，后面复用”。**

---

### 3. 新增了 `prewarm()`

位置：
- [image2video_fast.py](/home/hhhjyz/research/lingbot-world/wan/image2video_fast.py:150)

这是 fast 版一个很典型的工程优化点。

作用不是改结果，而是：
- 提前触发 CUDA kernel autotune
- 提前初始化 NCCL / FSDP / all-to-all
- 提前走一遍模型 shape

这样第一次正式 `generate()` 时，就不会把冷启动耗时算进去。

你可以把它理解成：
- base 版：直接跑
- fast 版：先热身，再正式跑

---

### 4. 推理是 chunked causal，而不是 full sequence

这是和 base 最大的机制差异。

base：
- 整段视频 latent 一次性丢给模型
- 每个 diffusion step 处理整段序列

fast：
- 视频按时间块切开
- 每次只处理一个 chunk
- 通过 KV cache 保留历史上下文

相关位置你后面重点看：
- [image2video_fast.py](/home/hhhjyz/research/lingbot-world/wan/image2video_fast.py:602)
- [image2video_fast.py](/home/hhhjyz/research/lingbot-world/wan/image2video_fast.py:608)

所以你读到 `chunk_size`、`current_start`、`kv_cache` 这类变量时，要立刻想到：
**“它正在把整段视频推理改成流式推理。”**

---

### 5. attention 机制变成 causal/local 可控

fast 版新增参数：
- `local_attn_size`
- `sink_size`
- `max_attention_size`

位置：
- [image2video_fast.py](/home/hhhjyz/research/lingbot-world/wan/image2video_fast.py:51)
- [generate_fast.py](/home/hhhjyz/research/lingbot-world/generate_fast.py:136)

它们对应的意义大致是：

- `local_attn_size`
  只看最近多少帧的上下文
- `sink_size`
  保留最早的一小部分“锚点 token”
- `max_attention_size`
  控制当前 attention 实际可见的最大 token 数

这说明 fast 版在主动限制注意力范围，换取更低时延和更小显存。

---

### 6. 采样流程和 base 不一样

base 版有：
- `sample_solver`
- `sampling_steps`
- 双模型
- 每一步都做 cond/uncond CFG

fast 版整体更偏固定化的快速采样路径。  
它仍然是扩散式生成，但为了实时和 chunked inference，流程组织更工程化。

你读的时候重点不是问“数学上是不是完全换了”，而是问：

- 哪些状态跨 chunk 保留了？
- 哪些状态每个 chunk 重置了？
- timestep 是怎么和 chunk 对齐的？

---

### 7. camera control 逻辑大体保留，但对齐方式更严格

camera 条件准备在 fast 版里仍然存在，位置：
- [image2video_fast.py](/home/hhhjyz/research/lingbot-world/wan/image2video_fast.py:499)

和 base 类似的部分：
- `intrinsics.npy`
- `poses.npy`
- `get_Ks_transformed`
- `interpolate_camera_poses`
- `compute_relative_poses`
- `get_plucker_embeddings`

不同点主要是：
- 需要按 `chunk_size` 和 `lat_f` 做更严格的时间裁剪/对齐
- camera embedding 后面要按 chunk 切开使用

也就是说：
**control 语义没大改，消费方式改了。**

---

## 推荐阅读提纲

你可以按这个顺序读，效率最高。

### 第一部分：先看它“想解决什么问题”
读这些位置：

- [image2video_fast.py](/home/hhhjyz/research/lingbot-world/wan/image2video_fast.py:34)
- [generate_fast.py](/home/hhhjyz/research/lingbot-world/generate_fast.py:174)
- [examples/persistent_inference.py](/home/hhhjyz/research/lingbot-world/examples/persistent_inference.py:1)

目标：
- 明确 fast 版不是单纯重写，而是为了低延迟和多次调用复用

---

### 第二部分：看初始化阶段新增了什么
重点看：

- `__init__`
  [image2video_fast.py](/home/hhhjyz/research/lingbot-world/wan/image2video_fast.py:34)

重点问题：
- 为什么只剩一个 `WanModelFast`？
- `pipe_dtype`、`local_attn_size`、`sink_size` 是干什么的？
- `_t5_cache` 和 `_cross_attn_initialized` 为什么要加？

---

### 第三部分：单独读 `prewarm()`
位置：

- [image2video_fast.py](/home/hhhjyz/research/lingbot-world/wan/image2video_fast.py:150)

重点问题：
- 它预热了哪些张量形状？
- 为什么 prewarm 不改变结果，只改变首轮时延？
- dummy latent / dummy c 2 ws / dummy context 分别在模拟什么？

这段很值得精读，因为它最能体现 fast 版的工程思路。

---

### 第四部分：对照 base 看前处理哪些相同，哪些不同
重点看 `generate()` 前半段，尤其是：

- 图像预处理
- latent shape 计算
- camera pose 处理
- T 5 编码缓存

位置大约从：
- [image2video_fast.py](/home/hhhjyz/research/lingbot-world/wan/image2video_fast.py:420)

重点问题：
- 哪些逻辑和 base 基本一样？
- 哪些地方为了 chunk 推理做了 shape 截断或对齐？

---

### 第五部分：重点攻克 chunk 推理主循环
这是最重要的部分。

你后面重点盯这些概念：
- `chunk_size`
- `kv_cache`
- `crossattn_cache`
- `current_start`
- `frame_seqlen`
- `max_attention_size`

这里你要搞清楚两件事：
- 一个 chunk 内部是怎么推理的
- chunk 和 chunk 之间怎么靠 cache 连起来

---

### 第六部分：最后再去模型侧验证
读这些文件：

- [wan/modules/model_fast.py](/home/hhhjyz/research/lingbot-world/wan/modules/model_fast.py:369)
- [wan/modules/model_fast.py](/home/hhhjyz/research/lingbot-world/wan/modules/model_fast.py:588)

重点问题：
- `c2ws_plucker_emb` 在 fast 版模型里怎么注入？
- causal self-attention 和普通 self-attention 的接口差异是什么？
- cache 是在哪层真正被消费的？

---

## 你可以带着这 6 个问题去读

1. fast 版为什么可以只用一个模型，而 base 要两个？
2. fast 版的“快”主要来自 cache，还是来自 chunk，还是来自 attention 截断？
3. camera pose 条件在 fast 版里有没有语义变化，还是只是喂法变了？
4. `chunk_size` 决定了什么？
5. `local_attn_size / sink_size / max_attention_size` 各自控制哪一层开销？
6. `prewarm()` 和 `persistent_inference.py` 体现了怎样的部署思路？

---

## 最后给你的阅读策略

不要从头逐行啃整份 `image2video_fast.py`。  
最省力的方法是：

1. 先拿 base 版当参照
2. 只标记 fast 版新增的变量、函数、cache、chunk 逻辑
3. 碰到和 base 一样的预处理，快速略过
4. 把主要精力放在：
- `__init__`
- `prewarm()`
- `generate()` 里的 chunk 主循环
- `model_fast.py` 的 cache 消费接口

如果你愿意，我下一步可以直接给你做一份“`image2video.py` vs `image2video_fast.py` 的函数级对照表”，把每个核心函数一一对应列出来。