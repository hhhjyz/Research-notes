# ActionCodec

**motivatioin**: 过去的 tokenizer 只关注动作重建的误差，却忽略了 token 质量对 VLA 网络本身优化过程的影响


high-performance action tokens 的四个标准：
- maximized temporal overlap between adjacent chunks,
- minimized vocabulary redundancy
- enhanced mutual information between tokens and multimodal contexts
- token independence
