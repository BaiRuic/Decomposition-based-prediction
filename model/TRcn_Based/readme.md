注意：为了便于统一接口,对初始模型做了以下修改
+ 将block 输入 由[batch_size, features, seqlen] 转成 [batch_size, seqlen, features]
+ 将block 输出 由[batch_size, features, seqlen] 转成 [batch_size, seqlen, features]
+ 将 stack 和 RRnn_Based里面的应该一样
