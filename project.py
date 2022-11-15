import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import SVC
from sklearn import svm,preprocessing
import time

data_BTC = pd.read_csv('BTC_history.csv')
data_DJI = pd.read_csv('DJI_sample.csv')

#移動平均線
data_BTC['MA5'] = data_BTC['Close'].rolling(5).mean()
data_BTC['MA20'] = data_BTC['Close'].rolling(20).mean()
data_BTC['MA60'] = data_BTC['Close'].rolling(60).mean()

#指數移動平均線
data_BTC['EMA12'] = data_BTC['Close'].ewm(span=12).mean()
data_BTC['EMA26'] = data_BTC['Close'].ewm(span=26).mean()

#MACD
data_BTC['DIF'] = data_BTC['EMA12'] - data_BTC['EMA26']
data_BTC['DEM'] = data_BTC['DIF'].ewm(span=9).mean()
data_BTC['OSC'] = data_BTC['DIF'] - data_BTC['DEM']

#作圖
fig,ax = plt.subplots(3,1,figsize=(10,10))
plt.subplots_adjust(hspace=0.8)
data_BTC['MA5'].plot(ax=ax[0])
data_BTC['MA20'].plot(ax=ax[0])
data_BTC['MA60'].plot(ax=ax[0])
data_BTC['EMA12'].plot(ax=ax[1])
data_BTC['EMA26'].plot(ax=ax[1])
data_BTC['Close'].plot(ax=ax[0])
data_BTC['Close'].plot(ax=ax[1])
ax[0].legend()
ax[1].legend()
data_BTC['DIF'].plot(ax=ax[2])
data_BTC['DEM'].plot(ax=ax[2])
ax[2].fill_between(data_BTC['Date'],0,data_BTC['OSC'])
ax[2].legend()
plt.show()

#相關係數
BTC_series = pd.Series(data_BTC['Close'])
DJI_series = pd.Series(data_DJI['Close'])
corr_BTC_DJI = round(BTC_series.corr(DJI_series), 2)

print("Corr of BTC&DJI:", corr_BTC_DJI)
plt.scatter(data_BTC['Close'], data_DJI['Close'])
plt.title("Corr of BTC&DJI:" + str(corr_BTC_DJI))
plt.show()

data_BTC.plot(x="Date", y="Close")
plt.show()

data_DJI.plot(x="Date", y="Close")
plt.show()

#SVM 預測比特幣走勢
df=data_BTC[['Close', 'Open', 'High', 'Low']]

data_BTC['diff'] = data_BTC["Close"]-data_BTC["Close"].shift(1)
data_BTC['diff'].fillna(0, inplace = True)

data_BTC['up'] = data_BTC['diff']  
data_BTC['up'][data_BTC['diff']>0] = 1
data_BTC['up'][data_BTC['diff']<=0] = 0

data_BTC['predictForUp'] = 0
target = data_BTC['up']
length=len(data_BTC)
trainNum=int(length*0.8)
PredictNum=length-trainNum

feature=data_BTC[['Close', 'Open', 'High', 'Low']]

feature=preprocessing.scale(feature)



featureTrain=feature[1:trainNum-1]
targetTrain=target[1:trainNum-1]
svmTool = svm.SVC(kernel='linear')
svmTool.fit(featureTrain,targetTrain)

predictedIndex=trainNum

while predictedIndex<length:
    testFeature=feature[predictedIndex:predictedIndex+1]           
    predictForUp=svmTool.predict(testFeature)   
    data_BTC.loc[predictedIndex,'predictForUp']=predictForUp   
    predictedIndex = predictedIndex+1

dfWithPredicted = data_BTC[trainNum:length]

figure = plt.figure()
   
(axClose, axUpOrDown) = figure.subplots(2, sharex=True)
dfWithPredicted['Close'].plot(ax=axClose,color="red")
dfWithPredicted['predictForUp'].plot(ax=axUpOrDown,color="red")
dfWithPredicted['up'].plot(ax=axUpOrDown,color="blue",label='Real Data')
plt.legend(loc='best')

major_index=dfWithPredicted.index[dfWithPredicted.index%2==0]
major_xtics=dfWithPredicted['Date'][dfWithPredicted.index%2==0]
plt.xticks(major_index,major_xtics)
plt.setp(plt.gca().get_xticklabels(), rotation=30)
plt.title("BTC ups and downs predicted by SVM")
plt.rcParams['font.sans-serif']=['SimHei']
plt.show()


#計算SVM勝率
a=0
for i in dfWithPredicted['predictForUp']:
    if(dfWithPredicted['predictForUp'][i]==dfWithPredicted['Up'][i]):
        a+=1
print(a)