import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_BTC = pd.read_csv('BTC_history.csv')
data_DJI = pd.read_csv('DJI_history.csv')
np_BTC = data_BTC.to_numpy()
np_DJI = data_DJI.to_numpy()

i = 0
j = 0
N = np_DJI.shape[0]

with open("result.txt", "w") as f:
    while(True):
        if j == N-1:
            break
        if(np_BTC[i][0] == np_DJI[j][0]) :
            i+=1
            j+=1
            continue
        else:
            tmp = np.copy(np_DJI[j-1]).tolist()
            tmp[0] = np_BTC[i][0]
            print(np.array(tmp))
            np_DJI = np.concatenate((np_DJI, np.array(tmp).reshape(1, 7)))
            # f.write(tmp)
            i+=1

def k(elem):
    return elem[0]

DJI = np_DJI.tolist()
DJI.sort(key=k)
np_DJI = np.array(DJI)

print(np_BTC.shape)
print(np_DJI.shape)
pd.DataFrame(np_DJI).to_csv('sample.csv')