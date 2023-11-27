import pandas as pd                # 데이터를 저장하고 처리하는 패키지
import matplotlib as mpl           # 그래프를 그리는 패키지
import matplotlib.pyplot as plt    # 그래프를 그리는 패키지
import numpy as np

import os
 
# csv 파일을 읽어서 DataFrame 객체로 만듦. 인덱스 컬럼은 point로 설정
this_program_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(this_program_directory)


dfScale1 = pd.read_csv('boxCounts(First).csv')
dataScale1 = dfScale1.astype(float)[30:-5].to_numpy().T[0]

dfScale2 = pd.read_csv('boxCounts(Second).csv')
dataScale2 = dfScale2.astype(float)[30:-5].to_numpy().T[0]


dfFirst = pd.read_csv('maskPixel(first).csv')
dfSecond = pd.read_csv('maskPixel(second).csv')

dataFirst = dfFirst.astype(float)[30:-6].to_numpy().T[0]
dataSecond = dfSecond.astype(float)[30:-5].to_numpy().T[0]





result1 = dataFirst/dataScale1
result2 = dataSecond/dataScale2

plt.plot(np.arange(len(result2)), result2, 'gray')
plt.plot(np.arange(len(result1)), result1, 'black')

plt.show()