import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import SVC
from sklearn import svm,preprocessing
import time

data_BTC = pd.read_csv('BTC_history.csv') #讀進2017/09/12-2022/09/12的BTC歷史資料
data_DJI = pd.read_csv('DJI_sample.csv')  #讀進2017/09/12-2022/09/12的DJI歷史資料

'''
#繪製BTC和DJI五年間的走勢圖
figure1 = plt.figure()
(BTCprice, DJIprice) = figure1.subplots(2, sharex=True)
data_BTC.plot(ax=BTCprice, x="Date", y="Close", label='BTC price')
data_DJI.plot(ax=DJIprice, x="Date", y="Close", label='DJI price')
plt.show()

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
'''

#SVM 預測比特幣走勢
df=data_BTC[['Close', 'Open', 'High', 'Low']]

#diff代表今日和前日收盤價的差值
data_BTC['diff'] = data_BTC["Close"]-data_BTC["Close"].shift(1)
data_BTC['diff'].fillna(0, inplace = True)

#up表示今日是否上漲，上漲表示1，下跌或平盤表示0
data_BTC['up'] = data_BTC['diff']  
data_BTC['up'][data_BTC['diff']>0] = 1
data_BTC['up'][data_BTC['diff']<=0] = 0

#predictForUp表示預測結果，同樣1表漲0表平與跌，先初始化為0
data_BTC['predictForUp'] = 0 

#target為實際之漲跌情況
target = data_BTC['up']
length=len(data_BTC)

#訓練集資料為原始資料的80%，預測集為20%
trainNum=int(length*0.8)
PredictNum=length-trainNum

#確認要進行訓練的特徵值，收盤價、開盤價、當日最高價、當日最低價
feature=data_BTC[['Close', 'Open', 'High', 'Low']]

#對特徵值進行標準化
feature=preprocessing.scale(feature)
featureTrain=feature[1:trainNum-1]
targetTrain=target[1:trainNum-1]

#kernel='linear', C=100 ->0.98 線性可分
#kernel='poly', degree=3, gamma='auto', C=50   ->0.76 高次方轉換
#kernel='rbf', gamma=0.7, C=50 ->0.92 高斯轉換
#gamma = 0.5、0.7、1、2 C=10^-1~10^2

svmTool = svm.SVC(kernel='linear', C=15)      #核函式取用linear kernel創建SVM分類器，linear kernel在特徵數量多時較為適用
svmTool.fit(featureTrain,targetTrain)   #通過fit函式，用特徵值和目標值訓練svmTool

predictedIndex=trainNum

#透過while逐一預測測試集
while predictedIndex<length:
    testFeature=feature[predictedIndex:predictedIndex+1]           
    predictForUp=svmTool.predict(testFeature)   
    data_BTC.loc[predictedIndex,'predictForUp']=predictForUp #將預測結果存放到data_BTC的predictForUp列中
    predictedIndex = predictedIndex+1

dfWithPredicted = data_BTC[trainNum:length] #只包含測試集

#作圖
figure2 = plt.figure()
(axClose, axUpOrDown) = figure2.subplots(2, sharex=True)
dfWithPredicted['Close'].plot(ax=axClose,color="blue", label='BTC price')
dfWithPredicted['predictForUp'].plot(ax=axUpOrDown,color="red", label='Predicted Data')
dfWithPredicted['up'].plot(ax=axUpOrDown,color="blue",label='Real Data')
plt.legend(loc='best')

major_index=dfWithPredicted.index[dfWithPredicted.index%2==0]
major_xtics=dfWithPredicted['Date'][dfWithPredicted.index%2==0]
plt.xticks(major_index,major_xtics)
plt.setp(plt.gca().get_xticklabels(), rotation=30)
plt.title("BTC ups and downs predicted by SVM")
plt.show()

accuracy = svmTool.score(featureTrain,targetTrain)
print("Accuracy:", accuracy)


#User input
                            
predict_next_day = svmTool.predict([['16433', '16204', '16543', '16088']]) #[['Close', 'Open', 'High', 'Low']]
print("SVM預測明日幣價漲跌: ", predict_next_day) #1代表上漲，0代表跌或平盤
