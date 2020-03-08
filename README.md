# 评论情感检测

该项目针对2020年美赛C题给出的评论数据(评论标题,评论内容)给出情感识别.

项目模型,数据引用[通过LSTM实现中英文语句情感分析，来判断消极还是积极](https://github.com/Elliottssu/lstm-sentiment-analysis)

## 项目说明

- `data`: 训练文本数据(实际上不需要训练了,但是需要用其构建词典)
- `body_pred`: 针对评论内容给出的positive的概率
- `headline_pred`; 针对评论标题给出的positive的概率
- `sentiment_analysis.py`: 主要程序,利用神经网络产生`body_pred`和`headline_pred`数据
- `main.py`: 根据`body_pred`和`headline_pred`综合考虑,利用设定的加权(headline占比0.2,body_pred占比0.8)生成综合的打分数据`pred`.
- `pred`: 百分制,小于50分即可看作`negative`,大于50分可看作`positive`,越大则越积极.
- `model_english.h5`: 已经训练好的神经网络