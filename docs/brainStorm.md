###首先不能从人类音乐家的认知系统来定义，因为人脑接收的 不光是声音传递的信息，
###还有多年经验积累的各种音节对照片段，在判别模型中很难做到积累如此惊人的噪点模型数据。


###其次，就仅有的少量标注数据来说，数据一定需要存在某个维度上的相似，才能
###得到良好的学习结果。就音乐的数学模型来看，应该是一定规则的音节的排列组合
###通过分析某个旋律片段的相似性，可以得到一个旋律相似度表，当然少量旋律的类似
###可以认为是某种程度的偶然或"致敬"，但是同之前的论文抄袭判别模型一样，超过
###一定百分比的雷同就应该视为不可接收的"原创"

###另一个难点在于即使同一个音乐创作者，在不同时期，不同作品中展示的风格有可能是
###完全迥异的，如果单就判断某个人来说，需要准备数据就要穷尽其所有的作品
###在某种程度上，尤其是作者还在世时，这个条件几乎是不可能达成的。

###所以另一个思路是仅仅比对两个相似作品是否为同一人所作。那么这个前提
###就是在音乐旋律上存在一些能被学习的特殊标记，展现出某一面的作曲习惯
###正如对文学作品分析其助词的使用频率而不在乎具体的故事情节和内容
###这个角度似乎能更好的解决判别模型的原始数据缺失问题


####那么问题来了，音乐作品中这些隐匿的判别标志是什么，又通过什么方式去学习呢？
#### 

