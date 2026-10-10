---
tags:
  - "#world_model"
status: active
arxiv: arXiv:2604.24764v1 [cs.CV] 27 Apr 2026
github: https://github.com/microsoft/World-R1.git
撰写时间: 2026-05-13
---
# World - R1

## Abstract

## Introduction

现有方法聚焦于 image-space generation, 缺少对 3D geometry 的理解， 在合成大范围相机移动或长时间驾驶场景时会出现几何幻觉和几何不一致性的问题， 物体可能变形、消失或非物理扭曲， 说明 _模型主要在学习图像空间相关性，而不是在模拟稳定的物理世界_

### Task and Application

bridge video generation with 3D representation

### Previous methods and technical challenges

#### 1

##### Previous method  
inject 3D priors into  inference process of video models
##### Failure case / limitation  
产生高昂的推理成本 限制模型的生成精度和泛化能力， 方法难以高效扩展， 也会牺牲模型原有的视频生成能力





