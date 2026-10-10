---
tags:
  - world_model
  - video_understanding
status: active
authors:
  - Wei Chen 等，香港科技大学（HKUST）与 Kling AI 
conference:
arxiv:
  - https://arxiv.org/abs/2609.35497v1
github:
  - https://github.com/HKUST-LongGroup/Sprout
撰写时间: 2026-10-09
---
# Sprout: Building Dynamic Memory While Reasoning For Agentic Video Understanding

---

## 1. Background & Overview 


### 1.1 Task & Application 
*   **Task:** Long video Understanding； 给定一段长视频 $V$ 和即将到来的多个问题 $q_1,q_2,\cdots,q_k$ ,系统需要回答每个问题，并能够利用之前观看视频时获得的信息
*   **Application:** 如何让 Video Agent 在有限的上下文和推理预算内，持续理解同一个长视频？

### 1.2 Previous Methods & Technical Challenges
*   **Challenge 1: offline memory construction 存在大量不必要的预处理开销** 
    *   *Previous method:* 
	    * **Hierarchical memories**，递归总结短的片段为多尺度摘要，实现 top-down retrieval
	    * **Graph-based memories**，建模 events/entities/time 之间的关系，在图结构中导航与遍历来回答问题
	    * **Flat episodic memories** 
    *   *Limitation:* 1. memory 在回答问题前基于整段视频离线构造，成本高；2. memory 是静态的，不会更新，模型的观察和结论不能跨问题持续
    *   *Technical reason:
	    * 当提出的问题很少时，预先构建是浪费的
	    * 不可更新的内存阻止了有用信息的积累
*   **Challenge 2: [概括技术痛点 2]**
    *   *(同上结构展开)*

### 1.3 Key Insight & Motivation 
*   **一句话概括:** 推理时在线构建 memory，memory 从粗糙到精细

### 1.4 Our Solution & Contributions 
Sprout，an agentic framework： memory 是一棵时间树，在回答问题时会滋生新的节点，通过 watch/remember/revisit 单个动作进行构建
- 模块/贡献 1：Watch — Question-driven Video Exploration
	- 解决痛点：完整视频预处理浪费计算资源。    
	- 做法：根据当前问题，从尚未观看的视频部分逐块进行低帧率观察。
	- 优势：只处理必要的视频片段，发现足够证据后可以提前停止。
    
- 模块/贡献 2：Remember — Dynamic Temporal Memory
	- 解决痛点：观察结果无法持久复用。
	- 做法：将已观察的视频转换为带时间戳的文本记忆节点，并建立时间树结构。
	- 优势：不同问题可以共享已积累的记忆。

- 模块/贡献 3：Revisit — Coarse-to-Fine Memory Refinement
	- 解决痛点：稀疏采样遗漏精细信息。
	- 做法：先从记忆中找到相关时间区间，再用更高帧率重新观察，并将新发现作为子节点写入。
	- 优势：只对相关片段增加视觉分辨率，避免对整个视频密集采样。
    
- 模块/贡献 4：Frame Releasing — Context-efficient Reasoning
	- 解决痛点：重复观察造成视觉 Token 在上下文中累积。
	- 做法：视频帧在被模型观察后，从后续的上下文历史中移除；保留文字观察结果，并允许重新访问原始视频。
	- 优势：减小后续模型调用的上下文规模，同时保留重新获取原始视觉证据的能力。

---

## 2. Method

### 2.1 Overview (全局总览)
*   **overview pipeline:** [给定 XX 输入，首先经过 XX，然后通过 XX，最终输出 XX。]

![[SPROUT-1791526184028.webp|697]]
![[SPROUT-1791547453208.webp]]
### 2.2 Module 1: [核心模块 1 名称]
*   **Motivation :** 为什么需要单独设计这个模块？
*   **Implementation :** 这个模块具体是怎么实现的？（公式、网络结构或核心算法）
*   **Why it works:**  它为什么能 work？它是如何约束或引导数据流的？
*   **Advantage:** 相比常规操作（如标准的 Attention 或卷积），它好在哪里？

### 2.3 Module 2: [核心模块 2 名称]
*   *(同上结构展开)*

---

## 3. Experiments 

### 3.1 Comparison Experiments 
*   **Baselines:** [对比了哪些核心 SOTA 方法？]
*   **Datasets:** [在哪些数据集上评估？]
*   **Metrics:** [评价指标是什么？重点关注哪个指标？]
*   **Core Conclusion:** [一句话总结结果，例如：在 XX 指标上超越 baseline，证明了该方法在 XX 方面（如姿态保持/生成多样性）的显著优势。]

### 3.2 Ablation Studies 
*   **[设计/模块 A] 消融:** 去掉该模块后，[某个具体指标/可视化效果] 变差，证明了它在 [具体作用] 上的必要性。
*   **[设计/模块 B] 消融:** *(同上)*

---

## 4. Limitation & Future Work 
> *批判性阅读，寻找方法的边界与自己的灵感。*

*   **Failure Cases (失败场景):** 在什么特定条件、数据或极端假设下，该方法会失效或表现不佳？
*   **Root Cause (原因分析):** 这种局限是由哪个模块的设计缺陷或前提假设导致的？
*   **My Thoughts (个人启发):** [针对上述局限，我可以怎么改进？或者这个方法的设计思路可以借鉴到我的哪个具体项目中？]