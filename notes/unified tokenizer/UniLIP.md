# UNILIP
两阶段训练方案和自蒸馏策略

希望构建一个**单一视觉表示**，同时服务于：
1. 多模态理解
2. 图像重建
3. 文生图
4. 指令图像编辑

CLIP 在这方面的问题是缺少像素细节

![[Pasted image 20260327135408.png]]

1. 给定 input image  $I$ , 先用 CLIP 提取视觉特征
	- 输出特征记为 $F_{clip}$
	- 空间大小是 $H/p \times W/p$
	- 通道维度是 $d$
2. pixel Decoder 和 CLIP 的特征维度不对齐，增加一个投影模块
$$ \hat I = D_{pix}(h_\phi(CLIP(I)))$$
3. **第一阶段训练**： 冻结 CLIP，训练 projection 和 pixel decoder， $\mathcal{L}_{stage1}=\mathcal{L}_{MSE}+\mathcal{L}_{LPIPS}$
4. **第二阶段训练**： 微调 CLIP，继续优化 projection 和 pixel decoder，增加 self-distillation approach, 防止 CLIP 偏离原始特征分布太多
![[Pasted image 20260327141032.png]]