# 评价对象的情感分析

## 预处理
* 使用gensim，中的word2vec对所有词进行word2vec处理，这里讲word变成50维，使用所有词训练10轮生成word vector保存在模型里
* 将句子使用0 vector 填充到相同长度

## 评价对象提取

* 将每个单词分为四类
1. O：非评价对象 
2. B：评价对象的第一个词
3. I：评价对象的后面几个词
4. X：对sentence进行pad的词

* 构造下面model，使用cross_entropy作为loss函数：

![alt text]( https://github.com/dark1412myj/IMageBase/blob/master/Semantic-Classification_1.jpg )

## 评价对象情感分析

* 将每个单词与评价对象进行拼接，形成100维的vector，使用两个lstm，分别处理评价对象前和后的词，将两个lstm的结果通过fc得到最终分析结果
* 构造下面model，使用cross_entropy作为loss函数：

![alt text]( https://github.com/dark1412myj/IMageBase/blob/master/Semantic-Classification_2.jpg )
