## Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos

### 1. Background
*   **Task and application**: 任务是让现有短视频 foundation model 在不重新训练、且不显著增加显存的前提下，生成无限长或超长视频，同时尽量保持主体、背景、运动和时序的一致性。应用场景包括电影制作、游戏开发、世界模拟、长叙事视频生成等，这些场景都要求视频不仅更长，而且在很长时间跨度上仍然连贯。
*   **Previous methods and technical challenges** :
    *   Challenge 1:
        *   Previous method: 一类 train-free 方法如 FreeNoise、FreeLong、FreePCA，通过增加输入 latent 数量、重组初始噪声或融合局部/全局信息，把短视频模型“拉长”到更长视频。
        *   Failure case / limitation: 这类方法虽然能扩展长度，但显存和计算会随着生成帧数线性增长，因此难以真正支持分钟级、无限帧视频。
        *   Technical reason: 它们本质上仍然要同时维护越来越长的 latent 序列，长程依赖建模是靠更大的输入上下文换来的，因此长度扩展与内存成本强耦合。
    *   Challenge 2:
        *   Previous method: FIFO-Diffusion 等 frame-level autoregressive 方法维护一个按噪声级别递增的 latent queue，通过出队一个已去噪 frame、入队一个新噪声 frame，实现固定内存的无限帧生成。
        *   Failure case / limitation: 尽管这类方法能做到常数内存，但生成的视频仍会出现内容漂移、局部伪影、主体/背景不稳定等问题。
        *   Technical reason: 训练时 foundation model 只见过“同一噪声级别”的一组 frame；推理时却必须同时处理“多个噪声级别”的 frame，造成严重的 training-inference gap。FIFO 虽然理论上分析了误差与噪声跨度相关，但实际输入给模型的局部窗口仍然跨越很大的噪声范围。
    *   Challenge 3:
        *   Previous method: FIFO-Diffusion 的一致性建模主要依赖局部窗口和相邻 chunk 的 lookahead denoising；一些 test-time scaling 方法则借助外部评估器做搜索或重采样。
        *   Failure case / limitation: 前者缺少显式的长程依赖，容易在长时间跨度上丢失主体/背景一致性；后者虽然能修正质量问题，但往往每一步都搜或者依赖外部模型，额外开销很大。
        *   Technical reason: 局部滑窗只能看见近邻帧，远距离 frame 之间的信息传播太弱；而外部评估器往往要求 clean pixel 输入，导致中间步骤还要额外 denoise 和 VAE decode，推理成本高且流程冗余。
*   **Our pipeline / solution**: 
    *   Key innovation: 提出 MIGA，用“两阶段训练-推理对齐 + 双重一致性增强”同时解决噪声错配和长期一致性问题，在 train-free、constant-memory 的前提下提升无限长视频质量。
    *   模块/贡献 1: Two-Stage Training-Inference Alignment (TTA) 解决“模型训练时只见统一噪声、推理时却见多噪声”的痛点；宏观上先用 zigzag queue 缩小噪声跨度，再在统一噪声层面做第二阶段去噪；优势是主动把推理输入分布拉回训练分布附近。
    *   模块/贡献 2: Dual Consistency Enhancement (DCE) 解决长期一致性不足的问题；宏观上对队尾高噪声 latent 做 self-reflection 早期纠错，对队首低噪声 latent 做 long-range frame guidance 显式引入远程上下文；优势是一头管“尽早发现并修正错误”，一头管“远距离帧持续通信”。
*   **Demos / applications**: 论文展示了基于 Wan2.1-1.3B 生成的三个 1000+ 帧长视频 demo，说明 MIGA 可以把原本默认仅支持 81 帧的基础模型扩展到无限帧生成；还展示了叙事长视频生成案例，支持多文本/分段文本控制。

### 2. Method
*   **Overview**:
    *   任务: 在不训练新模型的条件下，把短视频扩散模型扩展为无限长、长时一致的视频生成器。
    *   输入: 文本提示词；基础视频扩散模型；初始 Gaussian latent；一个按噪声级别组织的长 latent queue。
    *   输出: 长视频 latent 序列以及解码后的视频帧，长度可远超基础模型默认长度。
    *   整体流程: 先基于 frame-level autoregressive 生成框架维护一个 latent queue；第一阶段用 zigzag 结构让 queue 内噪声变化更平缓，逐批产生处于同一中间噪声级别的 latent；第二阶段把这些 latent 作为统一噪声级别的长序列继续滑窗去噪直到 clean；同时，在整个过程中对高噪声尾部做 self-reflection 异常检测与修正，对低噪声头部引入 long-range guidance，最终输出长视频。
    * ![[assets/Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos-1783596413982.webp]]
*   **Pipeline module 1: Frame-level autoregressive latent queue**
    *   Motivation: 如果直接把所有长视频帧一次性送进模型，显存会随长度增长，无法支持无限帧。
    *   做法: 维护长度为 denoising step 数 T 的 latent queue，队列中 latent 沿 frame 维具有递增噪声级别。每次对整个 queue 做一次 autoregressive inference，最前端 latent 被完全去噪并出队，同时在尾部补入新的 Gaussian noise latent。
    *   为什么能 work: 它把“视频长度”转化成“队列滚动次数”，模型每次只处理固定窗口，因此内存不随总帧数增长。
    *   Technical advantage: 相比 FreeLong/FreePCA 这类有限扩展方法，它真正支持 constant-memory 的无限帧生成。
*   **Pipeline module 2: TTA Stage 1 - Zigzag Iterative Denoising**
    *   Motivation: 标准 FIFO queue 中相邻 latent 噪声级别每帧都在变，导致滑窗输入覆盖的噪声跨度过大，训练-推理错配严重![[assets/Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos-1783601941027.webp]]。
    *   做法: 把 queue 设计成 zigzag 结构，不再每个 latent 都改变噪声级别，而是每隔 `Lzig` 个 latent 才改变一次。每次从队首弹出 `Lzig` 个处于同一噪声级别的 latent，再在队尾补入 `Lzig` 个纯噪声 latent。这个阶段只做部分去噪，队首 latent 还没到 clean。
    *   为什么能 work: 噪声级别变化变慢后，模型局部窗口看到的是更平滑、更窄的噪声分布，因此更接近训练时“统一噪声”的输入条件。
    *   Technical advantage: 它没有破坏 autoregressive 机制和常数内存特性，但主动缩小了 inference noise span，是对 FIFO 的直接改进。
*   **Pipeline module 3: TTA Stage 2 - Denoising at a Unified Noise Level**
    *   Motivation: 即使 Stage 1 缩小了噪声跨度，模型看到的仍不是完全统一的噪声级别；还需要进一步把推理条件贴近训练条件。
    *   做法: 经过若干次 Stage 1 后，累计得到 `n * Lzig` 个处于同一时间步 `τ(e-1)` 的 latent，把它们组成 `Qs2`。之后在这个统一噪声级别上对长序列继续做 sliding-window denoising，直到生成 clean video。
    *   为什么能 work: 在第二阶段，每次 denoising 输入的 latent 都具有相同噪声强度，这与模型训练范式一致，因此能显著减小错配带来的预测误差。
    *   Technical advantage: 相比只做单阶段 autoregressive 的方法，这一步相当于把“长视频生成”和“训练分布对齐”结合起来，同时仍保留固定显存。
*   **Pipeline module 4: DCE - Self-Reflection**
![[assets/Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos-1783598808316.webp]]
    *   Motivation: 长视频中的一致性问题如果等到后期 clean frame 才发现，纠错代价高；而现有基于外部评估器的 test-time scaling 又太慢。
    *   做法: 作者提出在 latent 空间中直接计算相邻 chunk 的 cosine similarity，作为一致性分数 `Cscore`。对 queue 尾部某个高噪声判断位置 `fjudg`，使用 `[f_judg-f_ref, f_judg - 1]` 作为参考，评估 `f_judg, f_judf+f_ref-1`， 如果相邻 chunk 的 ` Cscore ` 下降超过阈值 ` δadju `，就触发 expanded search。搜索时保留前面已经通过检查的 ` fguid ` 个 latent 作为 guiding context，对后续 latent 采样 ` nsamp ` 个候选并逐步去噪，最后选取一致性分数更高的候选替换原始片段。
    * ![[assets/Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos-1783599270808.webp|274x101]] 对每一帧 latent，在 token 维度上取平均,对每一帧的 c 维向量做归一化
    *   为什么能 work: 研究表明 latent 之间的距离反映了对应视频帧之间的差异程度。论文观察到，高噪声 latent 与最终 clean latent 的一致性曲线虽然绝对值不同，但波动趋势高度相关，所以可以在早期 noisy stage 就定位未来可能出现的一致性断裂，并及时修复。即使在较高的噪声强度（40）下 clean latents 和 noisy latents 之间也保持强烈的相关性
    * ![[assets/Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos-1783599847342.webp]]
    *   Technical advantage: 不需要 DINO 之类外部 evaluator，也不需要为了评估中间状态而频繁额外 denoise + decode，因而比 ScalingNoise 一类方法更轻量、更适合长视频 autoregressive 场景。
*   **Pipeline module 5: DCE - Long-Range Frame Guidance**
    *   Motivation: 滑窗去噪每次只能看局部 `f0` 个 latent，远距离帧之间缺乏直接交互，长期一致性容易退化。
    *   做法: 当模型处理当前位置 `l` 的局部窗口时，从更早的历史位置均匀稀疏采样 `mguid` 个较干净的 latent，并把它们拼接进当前 denoising 输入。对队列头部因历史不足而无法采样的情况，则不施加该指导。![[assets/Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos-1783601049882.webp|373x42]]
    *   为什么能 work: 这些更早的低噪声 latent 覆盖了更长的历史上下文，可以把远程时序信息直接注入当前局部去噪过程，从而抑制主体漂移和背景变化。
    *   Technical advantage: 实现简单，不需要改模型参数或重新训练，却显式弥补了纯局部滑窗在 long-range dependency 上的天然缺陷。
	* ![[assets/Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos-1783601104013.webp|452]]

### 3. Experiments 
*   **Comparison experiment**: 
    *   Baselines: 有限长度扩展方法包括 FreePCA、FreeLong；无限长度方法包括 FIFO-Diffusion、ScalingNoise；附录还与训练式长视频方法 CausVid、Self-Forcing、LongLive、Infinity-RoPE、Reward Forcing 做了讨论性比较。
    *   Datasets / tasks: 主实验在 VBench 和 NarrLV 上进行。VBench 评估长视频质量与一致性；NarrLV 评估长视频的 narrative expressiveness，并在 TNA=2/3/4 的设定下测试多段叙事能力。
    *   Metrics: VBench 采用 subject consistency、background consistency、motion smoothness、temporal flicker 和 overall score；NarrLV 采用 scene attributes、target attributes、target actions 三类指标。
    *   Results: 在 VBench 上，MIGA 对两种基础模型都达到最优或全面领先，说明它不仅适用于 VideoCrafter2，也能迁移到更强的 Wan2.1。相对 FIFO-Diffusion，作者强调其主体一致性和背景一致性分别提升 4.7% 和 2.0%，证明改进确实击中了 autoregressive 框架的核心短板，而不是只做了局部调参。相对 FreeLong 等有限扩展方法，MIGA 进一步提升一致性，说明“无限帧 + 高一致性”并不必然冲突。NarrLV 上，FIFO 本身已优于旧的 VideoCrafter2-based 方法，而 MIGA 又进一步提高，尤其 Wan2.1-based MIGA 在 richer narrative settings 下优势明显，说明更稳定的长期视觉内容能承载更复杂的叙事表达。定性上，作者展示了 1k 帧视频，说明方法在超长跨度下仍能保持主体与背景相对稳定。附录用户研究也显示，MIGA 在四个维度上都优于 FIFO-Diffusion。
*   **Ablation studies**:
    *   模块 A 消融: 去掉 TTA 和 DCE 后，baseline FIFO-Diffusion 的 O.S. 为 95.02；单独加入 TTA 后提升到 97.05，单独加入 DCE 后提升到 96.75，两者同时加入达到 97.82。这证明 TTA 是更核心的底层改进，DCE 则提供额外且互补的一致性增强。
    *   模块 B 消融: 在 TTA 内部，先加 Stage 1 再加 Stage 2，O.S. 从 95.02 依次提升到 96.78 和 97.05。说明 zigzag 结构先减少明显异常与错配，统一噪声的第二阶段再进一步抑制视频噪声并完成更强对齐。
    *   模块 C 消融: `Lzig` 从 1 增大时，整体性能先升后饱和，最佳在 `Lzig=4` 左右。证明“减小局部噪声跨度”确实有效，但跨度过窄后边际收益减弱。
    *   模块 D 消融: 如果只做 Stage 2、完全去掉 Stage 1，性能会显著下降。原因是这时不再是 autoregressive 信息传递，而是直接对独立初始化的长序列统一去噪，失去了 Stage 1 建立的隐式跨帧传播链。
    *   模块 E 消融: 调节 self-reflection 阈值 `δadju` 时，更小阈值会触发更频繁搜索，带来更高 correction rate 和更好的 O.S.，但 success rate 最终会下降，说明 test-time scaling 不是越多越好，而是存在收益-成本平衡点，作者最终选 `0.01`。
    *   模块 F 消融: `mguid` 从 0 增加后性能明显改善，最佳为 6，说明 long-range frame guidance 确实让远程历史信息对当前去噪有实质帮助，而不是冗余拼接。

### 4. Limitation
*   **失败场景**: 论文给出的坏例子中，一只猫在视频前半段运动正常，但到后面出现“头和尾突然对调”的严重结构性错误。说明在非常长的视频中，即便整体时序和外观一致性较强，模型仍可能出现物理结构违背、局部幻觉或对象拓扑错乱。
*   **原因分析**: 这类问题并不只是时序一致性弱，而更深层地来自基础视频生成模型缺乏稳定的物理知识与世界约束。MIGA 的 TTA 和 DCE 主要解决的是噪声分布错配与时序一致性传播问题，并没有从根本上注入 3D/物理先验，因此当视频足够长、误差逐步累积时，仍可能出现“看起来连续但本体结构错了”的 hallucination。
*   **未来改进方向**: 作者提出未来会引入除文本之外的额外条件信号，以生成更真实的长视频。进一步看，可以从三条线改进：一是引入结构化条件，如 depth、pose、layout、tracking 或 3D consistency；二是把 self-reflection 从“相似性纠错”扩展到“物理合理性纠错”；三是结合训练式 long-video tuning 或 world knowledge，以减少超长视频中的对象形变与物理违背。
