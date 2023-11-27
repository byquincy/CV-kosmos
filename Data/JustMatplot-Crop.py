import pandas as pd                # 데이터를 저장하고 처리하는 패키지
import matplotlib as mpl           # 그래프를 그리는 패키지
import matplotlib.pyplot as plt    # 그래프를 그리는 패키지
import numpy as np

import os
 
# csv 파일을 읽어서 DataFrame 객체로 만듦. 인덱스 컬럼은 point로 설정
this_program_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(this_program_directory)


dfCrop1 = pd.read_csv('crop_gunnerFarneback(first).csv')
dataCrop1 = dfCrop1.astype(int)[30:-5].to_numpy().T

dfCrop2 = pd.read_csv('crop_gunnerFarneback(second).csv')
dataCrop2 = dfCrop2.astype(int)[30:-5].to_numpy().T


print(dataCrop2)



dataCrop1[0] = (dataCrop1[0] + 15)
dataCrop2[0] = (dataCrop2[0] + 15)
plt.scatter(np.arange(len(dataCrop2[1])), dataCrop2[1], s=dataCrop2[0], c='black', alpha=0.7)
plt.scatter(np.arange(len(dataCrop1[1])), dataCrop1[1], s=dataCrop1[0], c='gray', alpha=0.7)

plt.show()