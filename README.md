## python机器学习（二）——美团评论二分类



这次的小项目就是对于美团的评论进行一个二分类的机器学习模型训练。

最后的效果就是输入一段评论，自动判断是好评还是差评。



还是一样的，由于是针对初学者，这篇文章会先介绍一点相关基础知识，再讲项目。



目录:

> pandas之DataFrame数据结构
>
> 文本特征提取——CountVectorizer
>
> 代码分析
>
> 源码展示



#### 一、pandas之DataFrame数据结构

___

##### 1.1 DataFrame介绍

DataFrame是一个表格型的数据结构，可以简单理解成excel。

比如下面这样：

```python
import pandas as pd
a={
	'name':['qx','cs','zy'],
    'id':['0001','0002','0003'],
    'sex':['male','male','female'],
}
b=pd.DataFrame(a)
print(b)
```

最后输出的结果就是下面这样:

|      | name | id   | sex    |
| ---- | ---- | ---- | ------ |
| 0    | qx   | 0001 | male   |
| 1    | cs   | 0002 | male   |
| 2    | zy   | 0003 | female |

我们在取值的时候可以这样:

```python
b['name']
```



##### 1.2 相关函数

接下来介绍两个相关函数iloc()和sort()

<font color='red'>iloc</font>可以取出对应的列

```python
b=b.iloc[:,[0,2]]
```

这就是把**所有行，第0列和第2列**取出来。

那么b现在就是

|      | name | sex    |
| ---- | ---- | ------ |
| 0    | qx   | male   |
| 1    | cs   | male   |
| 2    | zy   | female |



<font color='red'>sort</font>是进行排序，

```python
b=b.sort_values(axis=0,ascending=True, by=['name'])
```

其中

axis可以取0或者1:   0就是纵向排序,1是横向排序；

asending=True:   按照升序排列；

by=['name']:   按照名字进行排序。




#### 二、文本特征提取——CountVectorizer

___

先来看个小例子把，texts就是每个句子组合起来的列表。


```python
from sklearn.feature_extraction.text import CountVectorizer

texts=['I am handsome','handsome one is me','I am the one']
cv = CountVectorizer() # 创建词袋数据结构
cv_fit=cv.fit_transform(texts) # 把词语转换成词频矩阵

```

cv里面存储着词袋，cv_fit就是转换后的词频矩阵，是一个稀疏矩阵。

我们来看看它们里面的东西可能会更清楚一点。



先来看看词袋里有什么东西

```python
print(cv.get_feature_names())
#['am','handsome','is','me','one','the']
```

就是把出现过的词找到了，这就是词袋。

但是居然没有"I",可能是因为太短了吧。



接下来我们看看转换后的稀疏矩阵里面是什么样子。

```python
print(cv_fit.toarray())
#[[110000]
# [011110]
# [100011]]
```

很清晰了。

每一行就是一个句子，里面就是这个句子里面有没有出现过这个词。

所以实际上我们前一篇案例也可以用这个来做。



#### 三、代码分析

___

##### 3.1 读入数据和分析数据

```python
import pandas as pd
data=pd.read_csv("./train.csv",sep="\t",encoding='utf-8')
```

看看我们的data长什么样子

![](https://cdn.jsdelivr.net/gh/qxcs/qxBlogPicgo/img/20200709112718.png)

很清晰了哈，标签是label，文本信息是comment。

而且是只有两类标签，属于二分类的语料，完美。

还有些别的简单分析，我就不放出来了，在源码里面有。



##### 3.2 语料预处理

先做预处理，只取文本里的中文，会用到正则表达式，这个我就不具体展开了。

就是把非中文或者非字母数字的其他特殊符号，用空字符串替换。

```python
def remove_punctuation(line):
    line=str(line)
    if line.strip()=='':
        return ''
    rule=re.compile(r'[^a-zA-Z0-9\u4E00-\u9FA5]')
    line=rule.sub('',line)
    return line
data['clean_review']=data['comment'].apply(remove_punctuation)
```

定义的函数remove_punctuation()就是传进来的一串字符串，取出中文然后返回。

然后在data里面新生成一列，叫做'clean_review'。

现在来看看data['clean_review']长什么样子吧。

![](https://cdn.jsdelivr.net/gh/qxcs/qxBlogPicgo/img/20200709113636.png)

已经没有那些符号了。



那么下一步就是分词和干掉停用词。

会用到上一节讲过的lambda表达式哦

```python
def stopwordslist(filepath)
	stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords 

stopwords=stopwordslist('./cn_stopwords.txt')
data['clean_review']=data['clean_review'].apply(
    lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords])
)
```

先把停用词读取出来，这一步应该每有问题。

问题就是下一步，有点复杂。

apply()里面就是对data['clean_review']的每一行应用的函数，所以有一个lambda表达式。

那么这个函数在干嘛呢。

先看核心，**jb.cut(x)**,所以就是分词。

然后一个for循环，不在停用词里面的就返回。(这种for循环的写法我后面也会补充讲)

所以这个函数，把句子分词，然后把不在停用词里面的词语返回。



好了，接下来我们看一眼现在的data['clean_review']:

![](https://cdn.jsdelivr.net/gh/qxcs/qxBlogPicgo/img/20200709114347.png)

对比上一次的结果，可以发现是分好词了，也去掉了停用词的。

所以到此为止，我们的预处理已经搞定了。



##### 3.3 特征工程

但是用汉字是没有办法进行训练和预测的，需要变成数字构成的矩阵。

所以我们接下来要使用sklearn来进行特征工程，将每个评论变成一个矩阵，这样就可以方便的进行数学操作啦。

```python
from sklearn.feature_extraction.text import CountVectorizer

#先按照标签排序，因为后面要划分测试集和训练集
data=data.iloc[:,[0,2]]
data=data.sort_values(axis=0,ascending=True, by=['label'])
gEnd=data[data['label']==0].shape[0]
bEnd=data.shape[0]

#得到特征矩阵
x=list(data['clean_review'])
x=CountVectorizer().fit_transform(x)
y=list(data['label'])
```

不用过多关注前面的排序，那个是为了后面划分测试集。

主要是看到x的那里，用到了我们在开头讲过的CountVectorize()。

所以现在x是一个词频的稀疏矩阵，每一行代表一个评论；

而y是一个list，代表对应行的x的标签。



下一步是划分测试集和训练集，我们这里按照2:1的比例来划分。

```python
import math
import scipy.sparse as sp 

# 获得训练集的data
trainGX=x[0:math.ceil(gEnd*2/3)]
trainBX=x[gEnd:math.ceil((bEnd-gEnd+1)*2/3+gEnd)]
trainX = sp.vstack((trainGX,trainBX))
print(trainX.shape)

trainGY=y[0:math.ceil(gEnd*2/3)]
trainBY=y[gEnd:math.ceil((bEnd-gEnd+1)*2/3+gEnd)]
trainY=trainGY+trainBY
print(len(trainY))

# #接下来就是获得testData
testGX=x[math.ceil(gEnd*2/3):gEnd]
testBX=x[math.ceil((bEnd-gEnd+1)*2/3+gEnd):bEnd]
testX=sp.vstack((testGX,testBX))
print(testX.shape)
testGY=y[math.ceil(gEnd*2/3):gEnd]
testBY=y[math.ceil((bEnd-gEnd+1)*2/3+gEnd):bEnd]
testY=testGY+testBY
print(len(testY))
```

这一步应该也不难，就是把好标签取2/3出来，坏也取2/3出来，然后组装在一起，成为训练集。测试集则是取剩下的1/3。



唯一需要注意的就是稀疏矩阵的合并，和list的合并是不一样的。

用的是vstack()这个函数。



##### 3.4 选择分类器

至此，我们的处理全部搞定，可以选择机器学习的分类器开工了。

这里我们选择两种分类器分别进行训练：逻辑回归和朴素贝叶斯。



###### 3.4.1 逻辑回归(LogisticRegression)

我们先训练分类器

```python
from sklearn.linear_model import LogisticRegression
  
#训练LR分类器  
clf = LogisticRegression()  
clf.fit(trainX, trainY)
```

这代码。。嚯，真简单。

生成一个LR分类器，然后使用fit()函数，传入训练集就可以 了。



训练好了，就可以用了。。。。真是粗暴

```python
preY=clf.predict(testX)
```

原来预测就是预测(predict)啊,传上要预测的数据，也就是测试集就可以了。



最后我们再进行结果分析把，看看正确率有多高。

```python
clf.score(testX,testY) # 0.9330732292917167
```

看正确率，使用score()函数，传入测试集要  测试的样本  和  标注好的标签  就行了。

最后结果会是0.933，还是很不错的。



###### 3.4.2 朴素贝叶斯

```python
from sklearn.naive_bayes import MultinomialNB

# 训练
clf2 = MultinomialNB()
clf2.fit(trainX,trainY)
# 预测
y_pred2 = clf2.predict(testX)
# 结果分析
err2=sum(abs(y_pred2-testY))
total2=len(testY)
print("err:",err2,"  totalTest:",total2,"  errRatio:",err2/total2)
clf2.score(testX,testY) # 0.90906362545018
```

可以看到，使用流程基本和逻辑回归一样的，只是用了不同的包而已。

最后的正确率也是达到了0.909。



#### 四、源码展示

___

```python
import pandas as pd
data=pd.read_csv("./train.csv",sep="\t",encoding='utf-8')

###############
#停用词，分词处理
###############
import warnings
import jieba as jb
import pandas as pd
import re
warnings.filterwarnings("ignore")

def remove_punctuation(line):
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(r"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('',line)
    return line

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

data['clean_review'] = data['comment'].apply(remove_punctuation)
stopwords=stopwordslist('./cn_stopwords.txt')

data['clean_review'] = data['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))


###############
#特征工程
###############
from sklearn.feature_extraction.text import CountVectorizer

#先按照标签排序，因为后面要划分测试集和训练集
data=data.iloc[:,[0,2]]
data=data.sort_values(axis=0,ascending=True, by=['label'])
gEnd=data[data['label']==0].shape[0]
bEnd=data.shape[0]

#得到特征矩阵
x=list(data['clean_review'])
x=CountVectorizer().fit_transform(x)
y=list(data['label'])

#划分测试集和训练集
import math
import scipy.sparse as sp

# 获得训练集的data
trainGX=x[0:math.ceil(gEnd*2/3)]
trainBX=x[gEnd:math.ceil((bEnd-gEnd+1)*2/3+gEnd)]
trainX = sp.vstack((trainGX,trainBX))
trainGY=y[0:math.ceil(gEnd*2/3)]
trainBY=y[gEnd:math.ceil((bEnd-gEnd+1)*2/3+gEnd)]
trainY=trainGY+trainBY

# #接下来就是获得testData
testGX=x[math.ceil(gEnd*2/3):gEnd]
testBX=x[math.ceil((bEnd-gEnd+1)*2/3+gEnd):bEnd]
testX=sp.vstack((testGX,testBX))
testGY=y[math.ceil(gEnd*2/3):gEnd]
testBY=y[math.ceil((bEnd-gEnd+1)*2/3+gEnd):bEnd]
testY=testGY+testBY

###############
#机器学习模型
###############
#逻辑回归
from sklearn.linear_model import LogisticRegression
# 训练LR分类器
clf = LogisticRegression()
clf.fit(trainX, trainY)
# #进行预测
y_pred = clf.predict(testX)
# 结果分析
print('逻辑回归正确率:',clf.score(testX, testY))

#朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB

# 训练
clf2 = MultinomialNB()
clf2.fit(trainX,trainY)
# 预测
y_pred2 = clf2.predict(testX)
# 结果分析
print('朴素贝叶斯正确率:',clf2.score(testX,testY))
```

