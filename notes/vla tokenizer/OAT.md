# OAT

![[assets/Pasted image 20260326191558.png]]

1. 将 input action 和一组固定的 learnable register tokens $r_{1:H_l}$ 连接起来， transformer encoder 对他们一起处理，使 action token sequence 的信息能够聚合到 register 中
2. encoder 之后丢弃 action token 的部分


## Inducing Token Ordering

### Nested Dropout

给定长度为 $H_l$ 的 register token，随机采样 K， 尾部剩下的 token mask 掉

### Casual Attention

- 每个 register token 可以看到所有 action token
- register token 之间，只能看到在他前面的 register token


## Detokenization

不再让 decoder 直接一次性回归出最终动作，而是让它学会“如何把噪声逐渐变成真实动作“”

加噪的action sequence：![[assets/Pasted image 20260326201429.png]]

![[assets/Pasted image 20260326201911.png]]

![[assets/Pasted image 20260326201920.png]]

objective: $||\hat{v}-v||^2$
