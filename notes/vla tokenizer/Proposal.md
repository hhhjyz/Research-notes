# Proposal

## Datasets

仿真：
- LIBERO [[OAT]]
- RoboMimic [[OAT]]
- MetaWorld [[OAT]]
- RoboCasa [[OAT]]
- VLABench [[FASTer]]

Cross-embodiment:
- DROID
- BridgeData

## 结合离散和连续

### 参考 HART

分为 discrete tokens + residual tokens, discrete token 部分通过 VQ， residual token 部分通过一个小的 diffusioiin 模块

![[Pasted image 20260403111713.png]]
![[Pasted image 20260403111724.png]]

感觉可以在 OAT 上做一些修改，比如 register token 在经过 encoder 之后得到的 token 时连续的，那么此时的 register token = discrete token+ residual token

在生成阶段，先生成 discrete tokens, 接着 hidden states 经过 diffusion 模块生成 residual tokens

scalable-resolutioin AR 也自带从粗粒度到细粒度的属性，**大范围运动由 discrete token 表示，细节微调由 residual token 表示**，不过这样和 OAT 原来的从左到右的因果关系可能不太一样了

**或者可以前一半 token 是离散的，后面更细粒度的 token 用 residual token 来表示**


![[Pasted image 20260403113733.png]]



## 融合语义

### 参考 REPA/ActionCodec

中间层对齐

OAT 中的前几个 token 承载了 action chunk 的 global structure， 而 REPA 中证明在生成模型的早期/中间表征上，对齐外部强语义教师，训练会更快、最终效果更好

在 OAT tokenizer 训练时，除了原本的动作重建损失，再对前 K 个  token加入一个语义对齐损失

**考虑在量化前还是量化后去做对齐？**

**和什么去做对齐？** 前后帧状态的差，CLIP 提取的特征， ...

![[Pasted image 20260403140954.png]]

### 参考TokLIP

在保留动作重建能力的同时，增强语义的能力，加一个 casual attention，从 token 中提取语义，可以与视觉信息和文本提取的语义去进行对齐


![[Pasted image 20260403150420.png]]

### 参考 TokenFlow

参考这样的工作应该需要一个本身能提取比较强语义的 tokenizer 来做参考

**语义要从之前的 action 中提取吗，还是说只要依靠 视觉和文本的信息就够了？**

![[Pasted image 20260403152057.png]]