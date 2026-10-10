## TetherCache: Stabilizing Autoregressive Long-Form Video Generation with Gated Recall and Trusted Alignment

### 1. Background
*   **Task and application**: 研究自回归视频扩散模型的分钟级、开放时长视频生成。模型需要在固定显存与推理延迟预算下持续生成新帧，同时维持局部运动连续性、远距离语义一致性和长期画质稳定性；典型应用包括游戏模拟、交互式世界模型和交互电影。
*   **Previous methods and technical challenges**:
    *   Challenge 1: 固定 KV-cache 容量无法覆盖随视频长度线性增长的历史。
        *   Previous method: 标准局部缓存只保留最近的 $K$ 个 latent frame；sink-token、缓存压缩或检索类方法则固定保留早期帧，或依据时间近邻、注意力等单一信号选择历史帧。
        *   Failure case / limitation: 只保留最近帧时，早期用于校准注意力的稳定上下文以及远处的主体、场景和事件线索都会被丢弃；只按注意力选帧时，缓存又容易被同一短时间段内的相邻帧占满，远距离历史覆盖不足。
        *   Technical reason: 视频历史随 rollout 增长，而缓存预算固定，必须进行有损选择；单一 recency 或 attention criterion 没有显式惩罚时间冗余，因此“当前相关”不等于“在有限槽位中具有互补信息”。
    *   Challenge 2: 长期自条件生成造成 context distribution shift 和误差累积。
        *   Previous method: Teacher Forcing 以干净真实历史为条件；Self-Forcing 等方法通过短程 self-rollout 缩小训练与推理差距；SVI、Helios、Self-Forcing++、Rolling Forcing、HiAR 等通过训练时注入误差、延长训练窗口或联合多帧去噪增强鲁棒性。
        *   Failure case / limitation: 推理长度远超训练窗口后，模型持续消费自己的不完美输出，仍会出现噪声放大、颜色/光照漂移、物体变形、位置和语义漂移；训练式方案还需要重新训练模型。直接召回久远生成帧也可能把已经漂移的 K/V 特征重新注入上下文。
        *   Technical reason: 早期帧的条件分布最接近训练分布，而越晚生成的帧经历越多轮误差传递，其 K/V 的均值和标准差会逐渐偏离可靠上下文；异常 K/V 会改变 attention logit 与 softmax 的标定，即使内容语义相关，也可能污染后续注意力计算。
    *   Challenge 3: 历史信息保留与历史特征可信度是耦合问题。
        *   Previous method: 现有训练免费方法主要优化缓存布局、位置编码、压缩/检索或稀疏注意力；sink frame 通常只用于稳定注意力或位置外推。
        *   Failure case / limitation: 增加长程召回可以缓解遗忘，却会扩大漂移特征进入缓存的机会；只保留稳定 sink 又会牺牲中后段事件信息，无法表示随视频演化的场景状态。
        *   Technical reason: “该帧是否值得保存”是信息选择问题，“保存后的表示是否仍可信”是分布校准问题，两者需要不同机制；只解决其中一个无法同时兼顾长期覆盖与条件稳定性。
*   **Our pipeline / solution**:
    *   Key innovation: 提出训练免费、即插即用的 TetherCache，在不增加 KV-cache 容量和不修改模型参数的前提下，用“相关且多样的历史召回 + 可信统计对齐”同时解决历史覆盖不足与上下文分布漂移。
    *   模块/贡献 1: 将原始固定缓存划分为 Sink、Memory、Recent 三个功能区；Sink 保存最接近训练分布的初始帧，Recent 保证短期运动连续性，Memory 在固定槽位中提供选择性的远程上下文，从结构上解耦稳定锚点、局部连续性和长期记忆。
    *   模块/贡献 2: GRAB（Gated Recall with Attention-Diversity Balancing）让被 Recent 淘汰的帧与现有 Memory 共同竞争，综合当前 query 对候选帧的注意力相关性与时间多样性选出 top-$M$，避免纯局部缓存遗忘远处历史，也避免 attention-only 选择集中在相邻帧。
    *   模块/贡献 3: TAME（Trusted Alignment via Memory Editing）仅编辑新召回帧的 K/V，以 Sink 和已有 Memory 为 trusted pool，对其逐 head、逐 channel 的均值和标准差做部分对齐，在保留归一化内容模式的同时抑制漂移特征污染。
    *   模块/贡献 4: 方法保持原 KV-cache 张量形状和 attention 接口，只增加 memory 全局帧索引、sink 统计量及轻量选帧/编辑操作；结合未旋转 key 与 block-relative RoPE，移动记忆帧时不会破坏旋转位置。
*   **Demos / applications**: 论文展示最长 5 分钟的视频生成 Demo；定性案例包括昏暗房间内持续弹奏吉他、城市公园轮滑和烟花等场景。相比 Self-Forcing、$\infty$-RoPE、Deep Forcing，后段视频更少出现噪声、色偏、物体形变和主体语义漂移，并能长期保持场景布局、主体身份与运动连续性。

### 2. Method
*   **Overview**:
    *   任务: 在自回归扩散模型训练窗口之外持续生成分钟级视频，并在固定 KV-cache 预算下抑制长期画质与语义漂移。
    *   输入: 当前 AR step 的 noisy latent $z_i^{t_j}$、最终 clean context pass 的 query $Q_s$、当前帧 K/V token，以及由历史帧构成的固定容量缓存；文本 prompt 通过底层 Self-Forcing/Wan2.1 模型提供生成条件。
    *   输出: 当前新生成的 latent frame/chunk、更新后的 $S\Vert M\Vert R$ KV cache，以及供下一 AR step 使用的稳定历史上下文。
    *   整体流程: ① warm-up 阶段无条件写入新帧，缓存首次填满后冻结前 $S$ 帧为 Sink、中间 $M$ 帧为 Memory、末尾 $R$ 帧为 Recent；② 每个稳态 step 将新帧写入 Recent 尾部，并弹出 Recent 头部作为候选 $e_s$；③ GRAB 将 $e_s$ 与当前 Memory 合并，按注意力重要性和时间多样性重新打分并选 top-$M$；④ 若 $e_s$ 首次进入 Memory，TAME 以 Sink 和旧 Memory 为 trusted pool 分别校准其 K/V；⑤ 重写原物理 Memory 区域，读取时施加 block-relative RoPE，再对 $S\Vert M\Vert R$ 执行 attention。该过程逐 attention layer 独立运行。
*   **Pipeline module 1: Sink-Memory-Recent 三段式固定缓存**
    *   Motivation: 单一滑动窗口无法同时保存稳定初始上下文、短期连续运动和远程语义线索，而额外建立并行缓存会突破显存与延迟预算。
    *   做法: 将 $K$ 个 frame slot 原位解释为 $K=S+M+R$：Sink 冻结最初 $S$ 帧，Memory 保存 $M$ 个动态选择的中间历史帧，Recent 作为长度 $R$ 的 FIFO 滑动窗口。默认总预算 $K=21$，其中 $S=3$、$R=4$，因此 $M=14$。
    *   为什么能 work: 初始 AR 输出所依赖的条件最接近训练分布，适合作为稳定统计锚点；Recent 提供相邻帧条件以维持运动与外观连续；Memory 让有限缓存跨越更宽时间范围，恢复被局部窗口遗忘的主体和场景信息。
    *   Technical advantage: 不扩大原缓存、不修改 attention 输入形状，也不改变扩散采样器；只重分配现有槽位即可显式实现三类上下文功能。
*   **Pipeline module 2: GRAB 相关性-多样性门控召回**
    *   Motivation: Memory 容量有限，纯 FIFO 会遗忘长程内容，纯 attention 选择则偏好相邻且重复的高相关帧；缓存必须同时回答“对当前生成是否有用”和“是否增加历史覆盖”。
    *   做法: 在 rollout step $s$，构造候选池 $P_s=M_s^{cur}\cup\{e_s\}$，并重新评估旧 Memory 与新淘汰帧。对候选 $c$，将当前 clean query $Q_s$ 与候选 key $K_c$ 的点积在 head、query token、key token 上池化为标量 $\ell(c)$，再经候选间 softmax 得到重要性 $\phi_{imp}(c)$。时间冗余定义为
      $$
      r(c)=\max_{c'\neq c}\exp\left(-\frac{|g_c-g_{c'}|}{\sigma_s}\right)\phi_{imp}(c'),
      $$
      其中 $g_c$ 是全局帧索引，$\sigma_s$ 随候选时间跨度自适应；多样性奖励为 $\phi_{div}(c)=\max(0,1-r(c))$。最终用 $\phi(c)=\phi_{imp}(c)+\alpha\phi_{div}(c)$ 选择 top-$M$，默认 $\alpha=0.35$。
    *   为什么能 work: attention mass 直接复用模型当前 query-key 几何，衡量候选对当前预测的实际条件价值；重要帧附近的候选因高时间相似性获得较低多样性奖励，时间孤立帧获得更高奖励，因此缓存既保持语义相关性，又覆盖早期、中期和近期的不同历史段。
    *   Technical advantage: GRAB 不是 append-only：旧记忆会持续与新候选竞争，失效记忆可被降级；其轻量实现可先对 query 求和再做矩阵乘法，无需显式构造完整 $L\times |P_s|L$ attention tensor。
*   **Pipeline module 3: TAME 可信统计对齐**
    *   Motivation: GRAB 选中的远程帧可能本身来自已经漂移的 rollout；仅凭语义相关性无法保证其 K/V 在统计上适合作为未来条件。
    *   做法: 仅当淘汰帧 $e_s$ 新进入 Memory 时执行编辑。对其 key 或 value token $x\in\mathbb{R}^{L\times H\times D}$，以 $T_s=S\cup M_s^{cur}$ 为对应 K 或 V 的 trusted pool，沿 batch/token 维计算逐 head、逐 channel 的 $(\mu_x,\sigma_x)$ 与 $(\mu_T,\sigma_T)$。先得到完全对齐表示
      $$
      \widetilde{x}=\sigma_T\odot\frac{x-\mu_x}{\sigma_x}+\mu_T,
      $$
      再用 $\widehat{x}=(1-\tau)x+\tau\widetilde{x}$ 做部分编辑，默认 $\tau=0.6$；K 和 V 分开处理，Sink、Recent 与已接受的 Memory 不改动。
    *   为什么能 work: 仿射标准化保留召回帧内部的归一化内容结构，只把一、二阶统计拉回可信缓存分布；若候选本来可信，则 $\widetilde{x}\approx x$，编辑近似恒等，若候选漂移明显，则按 $\tau$ 将其拉回稳定区域。
    *   Technical advantage: 相比丢弃漂移帧或重训模型，TAME 是局部、保守且训练免费的 token 编辑；trusted pool 同时包含稳定 Sink 和缓慢演化的 Memory，比只使用固定早期分布更能兼顾长期场景变化。
*   **Pipeline module 4: 位置处理与层内执行**
    *   Motivation: 被召回帧会被移动到物理 Memory slot；若直接保留原旋转后的 key 或按物理槽位解释位置，RoPE 相位与真实时间关系可能失配。
    *   做法: 缓存未旋转的 key，并保存 Memory 中每帧的全局索引；从缓存读取时再施加 block-relative RoPE。扩散去噪过程中同一 recent slot 可被多次 noisy pass 覆写，但 GRAB/TAME 只在最终 clean context pass 上利用 clean query 执行一次。
    *   为什么能 work: 将内容缓存与位置旋转解耦，避免帧在缓存区域间迁移时破坏位置编码；只在 clean pass 选帧也避免用含噪 query 做不稳定决策。
    *   Technical advantage: 维持 baseline 的缓存预算和 attention 接口；额外开销限于每帧、每层一次小候选打分以及新帧被接纳时的统计对齐，实测相对 Self-Forcing 延迟增加不足 6%。

### 3. Experiments
*   **Comparison experiment**:
    *   Baselines: 以 Self-Forcing 为直接 backbone/baseline，并比较 CausVid、$\infty$-RoPE、MemRoPE 和 Deep Forcing；其中既包含 AR 生成基线，也包含训练免费的长上下文位置/缓存管理方法。
    *   Datasets / tasks: 从 MovieGen-Bench 随机抽取 128 个 prompt 生成 30 秒和 60 秒视频，另抽取 32 个 prompt 生成 240 秒视频；prompt 经 Qwen2.5-7B-Instruct 改写，分辨率为 $832\times480$，使用 VBench-Long/VBench 评估。人评由 14 名参与者按 2AFC 比较同 prompt 视频。
    *   Metrics: Dynamic Degree、Imaging Quality、Color Consistency、Overall Consistency、VBench Total/Quality/Semantic Score；$\Delta$Quality Drift 定义为视频首尾 clip 的 Imaging Quality 差值，用于刻画误差累积，越低越好；人评考察主体一致性、成像质量与整体质量偏好。
    *   Results: TetherCache 在 30、60、240 秒三种时长均取得最高 Total Score，说明缓存管理没有牺牲短视频性能，且时长越长优势越明显。240 秒时，Self-Forcing 的 Semantic Score 降至 44.34、$\Delta$Quality Drift 达 7.84；TetherCache 分别达到 74.85 和 1.33，并取得最高 Imaging Quality、Overall Consistency 与 Total Score，直接支持“长程召回减少语义遗忘、可信对齐抑制画质漂移”的核心主张。相较各基线，人评平均偏好率均超过 66%，表明自动指标提升能转化为可感知的主体一致性和视觉质量改善。定性结果中，基线后段出现噪声、色偏、物体变形和语义弱化，而 TetherCache 更稳定地保持场景布局、主体身份与动作连续性。
*   **Ablation studies**:
    *   GRAB 消融: 在 240 秒设定中，Self-Forcing 加入 GRAB 后，Total Score 从 71.22 提升到 78.80，Semantic Score 从 44.34 提升到 71.05，$\Delta$Quality Drift 从 7.84 降到 3.64；说明相关性-多样性召回在固定预算下恢复了有效长程条件，并显著减轻语义退化。
    *   TAME 消融: 在 GRAB 基础上加入 TAME 后，Total Score 进一步升至 80.17，Semantic Score 升至 74.85，$\Delta$Quality Drift 降至 1.33，且全部报告指标一致改善；烟花案例中 GRAB 已改善色彩一致性但仍有局部伪影，加入 TAME 后伪影消失，证明“选对记忆”之外还必须校准其分布。
    *   GRAB 多样性项分析: attention-only 的保留图集中在窄对角带，表明缓存快速遗忘远处帧；加入 diversity 后出现跨多个 AR step 持续存在的水平保留轨迹，Memory 覆盖多个时间段。不同 $\alpha$ 下整体较稳定，$\alpha=0.35$ 最佳；过小会退化为时间聚集的 attention 选择，过大则可能为分散性接纳相关性不足的帧。
    *   TAME 强度分析: 30 秒生成对 $\tau$ 不敏感，符合短 rollout 漂移较小的预期；240 秒性能随 $\tau$ 增大而提升并在约 0.6 饱和，说明统计校准主要服务于长期误差累积。过强编辑不再增益，可能压制有用的内容特异变化。
    *   缓存分区消融: 固定总预算下，将 Sink 从 0 增至 3 可提升性能，验证可信早期统计的价值；增至 6 后因挤占 Memory 容量而下降。Recent 取 4 最优，更大的 Recent 会压缩长程记忆槽位，证明三类缓存之间存在稳定先验、局部连续性与远程覆盖的容量权衡。
    *   漂移与效率分析: Self-Forcing 的 latent 均值和标准差偏移随帧索引单调增长，$\infty$-RoPE 只能减缓，而 TetherCache 在初始上下文后保持低且稳定的偏移，直接验证 TAME 的机制。240 秒生成中参数量仍为 1.3B，延迟从 Self-Forcing 的 1126.1 秒增至 1192.5 秒、吞吐为 3.22 FPS，开销不足 6%，说明提升并非依靠扩大模型或缓存预算。

### 4. Limitation
*   **失败场景**: 论文未单列失败案例，以下为由实验边界和方法机制推断的风险：① TAME 只校准 K/V 的逐通道均值与标准差，面对几何结构错误、主体替换、因果事件错误等高阶语义漂移时可能无能为力；② GRAB 的候选池仅由当前 Memory 与刚被 Recent 淘汰的一帧组成，一旦重要帧早期未被接纳或被淘汰，之后无法从完整历史中重新检索；③ 固定 $S/M/R$、$\alpha$ 和 $\tau$ 在镜头切换、快速场景演化或交互动作分布变化时未必最优；④ 证据主要来自基于 Wan2.1 的 Self-Forcing、MovieGen-Bench prompt 和 VBench，尚未证明对其他 backbone、真实交互控制或更长于 5 分钟 rollout 的泛化；⑤ 虽然相对开销低于 6%，240 秒视频仍需约 20 分钟生成，离实时世界模拟有明显距离。
*   **原因分析**: TAME 建立在“一、二阶统计偏移是长期污染的重要表征，Sink/已有 Memory 可作为可信分布”的假设上；当早期 Sink 与当前场景差异很大，或已有 Memory 也已被污染时，trusted pool 不再完全可信，统计拉回还可能压制合理的分布变化。GRAB 是在线、不可逆的有限候选选择，缺少外部历史库和语义级检索，因此不能恢复已经彻底丢弃的信息。固定分区与固定超参数也无法根据场景变化速度、当前不确定度和漂移强度动态分配缓存。
*   **未来改进方向**: 可引入可检索的低成本外部历史摘要或分层记忆，使被淘汰内容仍可重召回；用场景变化、生成置信度和在线漂移检测动态调整 $S/M/R$、$\alpha$ 与 $\tau$；将 TAME 从均值/方差匹配扩展到协方差、低秩子空间或语义一致性约束，并为 trusted pool 增加可信度估计和异常剔除；在多 backbone、动作条件交互世界模型、镜头切换数据和更长时长上验证，同时结合 KV 压缩/稀疏注意力降低绝对推理成本。
