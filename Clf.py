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