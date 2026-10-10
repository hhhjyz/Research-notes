---
tags:
  - world_model
status: active
arxiv: arXiv:2503.20314v2
github:
撰写时间: 2026-05-24
---
**学习重点**  
这周最值得抓的主线，其实可以拆成两条：

1. `Wan2.1 -> Wan2.2` 在 `T2V / I2V / TI2V` 条件注入方式上的演进
2. `camera pose / c2w / plucker embedding / PRoPE / shift-scale` 要单独成一条“扩展阅读线”

原因是我查了你给的官方代码，`Wan2.1` 和 `Wan2.2` 的主仓里，核心生成路径里没有明显的 `camera pose / c2w / plucker / PRoPE` 实现；主线还是“文本条件 + 图像条件 + 掩码/latent 注入 + RoPE 时空位置编码”。

你这周如果时间有限，建议先把 Wan 主线吃透，再把 camera pose 那条单独挂出去。

**先给结论**  
你可以按下面这个 6 天计划学，每天目标都尽量落到具体文件和“产出问题”。

**Day 1: 先建立总图，搞清楚每个模型到底负责什么**  
目标：把模型家族关系背熟，不进细节。

看这些文件：

- [Wan2.1/generate.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/generate.py)
- [Wan2.2/generate.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/generate.py)
- [Wan2.1/wan/configs/wan_t2v_1_3B.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/configs/wan_t2v_1_3B.py)
- [Wan2.1/wan/configs/wan_t2v_14B.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/configs/wan_t2v_14B.py)
- [Wan2.1/wan/configs/wan_i2v_14B.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/configs/wan_i2v_14B.py)
- [Wan2.2/wan/configs/wan_t2v_A14B.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/configs/wan_t2v_A14B.py)
- [Wan2.2/wan/configs/wan_i2v_A14B.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/configs/wan_i2v_A14B.py)
- [Wan2.2/wan/configs/wan_ti2v_5B.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/configs/wan_ti2v_5B.py)

当天要记住：

- `Wan2.1`：
    - `T2V-1.3B` 是小模型版 T 2 V
    - `T2V-14B` 是大模型版 T 2 V
    - `I2V-14B` 是单图到视频
    - 还有 `FLF2V`，就是 first-last-frame 到视频
- `Wan2.2`：
    - `T2V-A14B` 和 `I2V-A14B` 是 MoE 双专家版本
    - `TI2V-5B` 是统一框架，同时支持 T 2 V 和 I 2 V

当天产出：

- 自己写一页表格：`模型名 / 输入 / 条件形式 / VAE / 是否双专家 / 默认shift / 默认guide scale`

**Day 2: 吃透 Wan 2.1 的 T 2 V 主路径**  
目标：明白最基础的 text-to-video 是怎么跑起来的。

重点看：

- [Wan2.1/wan/text2video.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/text2video.py)
- [Wan2.1/wan/modules/model.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/modules/model.py)

重点问题：

- 文本条件怎么来
- latent shape 怎么定
- timestep / scheduler / CFG 怎么接进去
- transformer 里 RoPE 是怎么加的

关键实现点：

- T 5 文本编码在 `WanT2V.generate`
- CFG 是 `cond/uncond` 两次前向再相减相加
- 时空 RoPE 在 [Wan2.1/wan/modules/model.py (line 32)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/modules/model.py:32) 和 [Wan2.1/wan/modules/model.py (line 43)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/modules/model.py:43)
- block 主干在 [Wan2.1/wan/modules/model.py (line 238)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/modules/model.py:238)
- 整体 forward 在 [Wan2.1/wan/modules/model.py (line 493)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/modules/model.py:493)

当天产出：

- 画一张数据流图：`prompt -> T5 -> context -> DiT -> VAE decode`

**Day 3: 吃透 Wan 2.1 的 I 2 V / FLF 2 V，重点学“图像条件怎么注入”**  
目标：搞清楚 Wan 不是怎么“理解图像”，而是怎么把图像约束塞进视频生成。

重点看：

- [Wan2.1/wan/image2video.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/image2video.py)
- [Wan2.1/wan/first_last_frame2video.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/first_last_frame2video.py)
- [Wan2.1/wan/modules/model.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/modules/model.py)

这一天最重要的几个点：

- I 2 V 不是只靠 cross-attn 接图片
- 它其实有两路图像条件：
    - CLIP 图像特征接到文本上下文前面
    - 首帧 VAE latent 加 mask，直接跟噪声 latent 在 channel 维拼接

代码证据：

- mask 和首帧 latent 构造在 [Wan2.1/wan/image2video.py (line 210)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/image2video.py:210) 到 [Wan2.1/wan/image2video.py (line 249)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/image2video.py:249)
- 条件前向参数在 [Wan2.1/wan/image2video.py (line 284)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/image2video.py:284)
- `clip_fea + y` 真正送进模型在 [Wan2.1/wan/modules/model.py (line 523)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/modules/model.py:523) 到 [Wan2.1/wan/modules/model.py (line 563)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/modules/model.py:563)
- I 2 V cross-attn 把 image tokens 和 text tokens 分开算，再相加，在 [Wan2.1/wan/modules/model.py (line 187)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/modules/model.py:187)
- FLF 2 V 把首尾帧一起作为条件，在 [Wan2.1/wan/first_last_frame2video.py (line 231)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/first_last_frame2video.py:231) 和 [Wan2.1/wan/first_last_frame2video.py (line 262)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/first_last_frame2video.py:262)

当天要形成的理解：

- `T2V`：只有 text context
- `I2V`：`text context + image context tokens + masked image latent`
- `FLF2V`：`text context + first/last frame image tokens + first/last frame latent anchors`

**Day 4: 学 Wan 2.2 的 A 14 B，重点抓“MoE 怎么改变 T 2 V/I 2 V”**  
目标：别只记“Wan 2.2 更强”，而是要知道它具体强在哪个实现决策上。

重点看：

- [Wan2.2/wan/text2video.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/text2video.py)
- [Wan2.2/wan/image2video.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/image2video.py)
- [Wan2.2/wan/configs/wan_t2v_A14B.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/configs/wan_t2v_A14B.py)
- [Wan2.2/wan/configs/wan_i2v_A14B.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/configs/wan_i2v_A14B.py)

核心变化：

- `Wan2.2 A14B` 不再是单个 DiT，而是 `low_noise_model + high_noise_model`
- 不同 timestep 切不同专家
- `guide_scale` 也可以按高噪声/低噪声分开设

代码证据：

- 双模型加载在 [Wan2.2/wan/text2video.py (line 89)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/text2video.py:89) 和 [Wan2.2/wan/text2video.py (line 96)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/text2video.py:96)
- timestep 选择专家在 [Wan2.2/wan/text2video.py (line 152)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/text2video.py:152)
- `boundary` 配置在 [Wan2.2/wan/configs/wan_t2v_A14B.py (line 35)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/configs/wan_t2v_A14B.py:35) 和 [Wan2.2/wan/configs/wan_i2v_A14B.py (line 35)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/configs/wan_i2v_A14B.py:35)
- I 2 V 里按 boundary 切换 guide scale 在 [Wan2.2/wan/image2video.py (line 388)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/image2video.py:388)

当天要回答的问题：

- 为什么 T 2 V 默认 `sample_shift=12.0`，I 2 V 默认 `5.0`
- 为什么 T 2 V 的 `guide_scale=(3.0, 4.0)`，I 2 V 是 `(3.5, 3.5)`

**Day 5: 学 Wan 2.2 的 TI 2 V-5 B，重点抓“统一框架”**  
目标：真正理解你记的“始终把第一帧替换成图像”这句话。

重点看：

- [Wan2.2/wan/textimage2video.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/textimage2video.py)
- [Wan2.2/wan/configs/wan_ti2v_5B.py](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/configs/wan_ti2v_5B.py)

核心结论：

- `TI2V-5B` 是一个统一模型，不再分开 T 2 V / I 2 V 模型类
- `generate()` 里根据 `img is None` 分流到 `t2v()` 或 `i2v()`，在 [Wan2.2/wan/textimage2video.py (line 162)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/textimage2video.py:162)
- `i2v()` 里不是 CLIP + extra context 那套，而是直接把首帧 latent 写死进生成过程
- 每一步采样后又把首帧区域替换回去，这就是你记的“始终把第一帧替换成图像”

代码证据：

- 首帧编码在 [Wan2.2/wan/textimage2video.py (line 512)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/textimage2video.py:512)
- 初始 latent 用 mask 混合原图 latent 在 [Wan2.2/wan/textimage2video.py (line 549)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/textimage2video.py:549) 到 [Wan2.2/wan/textimage2video.py (line 552)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/textimage2video.py:552)
- 每一步采样后再替换一次在 [Wan2.2/wan/textimage2video.py (line 598)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/textimage2video.py:598)
- T 2 V 路径里还用了 `mask1, mask2` 生成逐 token timestep，在 [Wan2.2/wan/textimage2video.py (line 358)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/textimage2video.py:358) 和 [Wan2.2/wan/textimage2video.py (line 373)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.2/wan/textimage2video.py:373)

这一天要形成一句自己的总结：

- `Wan2.1 I2V` 更像“图像 token + 图像 latent 双条件”
- `Wan2.2 TI2V-5B I2V` 更像“统一视频生成模型 + 首帧强约束重注入”

**Day 6: 单独整理 camera pose / PRoPE / c 2 w / plucker 线**  
目标：避免把“官方 Wan 主线”和“可控相机/位姿扩展”混掉。

我在你这两个官方仓库里没有找到以下关键词的核心实现：

- `camera pose`
- `c2w / c2ws`
- `plucker / pluker embedding`
- `PRoPE`

但找到了标准 `RoPE`：

- [Wan2.1/wan/modules/model.py (line 32)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/modules/model.py:32)
- [Wan2.1/wan/modules/model.py (line 43)](/Users/hhhjyz/Library/CloudStorage/OneDrive-个人/Code/research/Wan2.1/wan/modules/model.py:43)

所以这一天建议你做的不是“硬啃官方代码”，而是整理一个问题清单：

- `PRoPE` 是不是某篇论文里的 pose-aware rotary embedding
- `c2w + plucker embedding` 是不是把 camera ray / pose 编成 token 或 positional bias
- 它和 Wan 的原生时空 RoPE 是“替换关系”还是“附加条件关系”
- `shift / scale` 在你记的 camera pose 语境里，是不是 pose normalization，而不是 scheduler shift

**你这周的最终产出建议**  
学完后，至少写出这 3 页笔记：

1. `Wan2.1 vs Wan2.2` 模型族对照表
2. `T2V / I2V / TI2V` 条件注入机制对照图
3. `官方 Wan 主线` 与 `camera pose 扩展线` 的边界说明

**一个你现在就可以用的对照结论**  
最核心的差异可以先记成这几句：

- `Wan2.1 T2V`
    - 纯文本条件
    - 单模型 DiT
    - 标准时空 RoPE
- `Wan2.1 I2V`
    - 文本 + CLIP 图像 token + 首帧 VAE latent/mask
    - 图像条件既进 cross-attn，也进输入通道
- `Wan2.1 FLF2V`
    - I 2 V 的双端点版本
    - 首尾帧都做 anchor
- `Wan2.2 T2V-A14B / I2V-A14B`
    - 主要升级是 MoE 双专家
    - 高噪声阶段和低噪声阶段分开建模
- `Wan2.2 TI2V-5B`
    - 统一 T 2 V/I 2 V
    - I 2 V 通过首帧 latent 持续回填来强约束第一帧
    - 不再走 Wan 2.1 那种 CLIP image token 路线

**建议你怎么分配时间**  
如果这周只有 3 天可用，就压缩成：

1. 先学 `Wan2.1 T2V + I2V`
2. 再学 `Wan2.2 A14B + TI2V-5B`
3. 最后单独梳理 `camera pose / PRoPE` 问题清单

如果你愿意，我下一步可以直接帮你做一份：

1. “按文件跳转”的精读清单
2. 或者一张 `Wan2.1 / Wan2.2 / T2V / I2V / TI2V` 对比表格