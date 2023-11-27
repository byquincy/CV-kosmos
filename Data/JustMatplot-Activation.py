import pandas as pd                # 데이터를 저장하고 처리하는 패키지
import matplotlib as mpl           # 그래프를 그리는 패키지
import matplotlib.pyplot as plt    # 그래프를 그리는 패키지
import numpy as np

import os
 
# csv 파일을 읽어서 DataFrame 객체로 만듦. 인덱스 컬럼은 point로 설정
this_program_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(this_program_directory)

dfFirst = pd.read_csv('maskPixel(first).csv')
dfSecond = pd.read_csv('maskPixel(second).csv')

dataFirst = dfFirst.astype(int)[5:-5]
dataSecond = dfSecond.astype(int)[5:-5]

plt.plot(np.arange(len(dataFirst)), dataFirst, 'black')
plt.plot(np.arange(len(dataSecond)), dataSecond, 'gray')
plt.show()