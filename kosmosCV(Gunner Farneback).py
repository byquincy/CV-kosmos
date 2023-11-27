import cv2
import numpy as np
import csv
from tqdm import tqdm

csvLine = []

# FILE = 'c-elegance_5x.mp4'
# FILE = 'second_5x.mp4'
FILE = 'first_5x.mp4'

datas = []

def drawFlow(img,flow,step=16):
    global datas

    h,w = img.shape[:2]
    # 16픽셀 간격의 그리드 인덱스 구하기 ---②
    idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(np.int32)
    indices =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,2)
    
    for x,y in indices:   # 인덱스 순회
        if abs(np.sum(flow[y, x].astype(np.int32))) <= 2:
            continue

        # 각 그리드 인덱스 위치에 점 그리기 ---③
        cv2.circle(img, (x,y), 1, (0,255,0), -1)
        # 각 그리드 인덱스에 해당하는 플로우 결과 값 (이동 거리)  ---④
        dx,dy = flow[y, x].astype(np.int32)
        # 각 그리드 인덱스 위치에서 이동한 거리 만큼 선 그리기 ---⑤
        cv2.line(img, (x,y), (x+dx, y+dy), (0,255, 0),2, cv2.LINE_AA )

        # 자체 분석용
        datas[-1][0] += dx
        datas[-1][1] += dy



vidcap = cv2.VideoCapture(FILE)

count = 0

prev = None
for temp in tqdm(range(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)):
    success,image = vidcap.read()
    if not success: break
    
    opticalFlow = np.zeros((960, 1280, 3), np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if prev is None:
        prev = gray
    else:
        flow = cv2.calcOpticalFlowFarneback(prev,gray,None,\
                0.5,3,15,3,5,1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN) 
        datas.append([0, 0])
        drawFlow(opticalFlow, flow)
        prev = gray

    cv2.imshow('OPTICAL_FLOW', opticalFlow)
    cv2.imwrite('OPTICAL_FLOW'+str(temp)+".png", opticalFlow)
    if cv2.waitKey(1) == 27:
        break

    if not success:
        break

cv2.destroyAllWindows()

# with open('gunnerFarneback.csv', 'w') as f:
#     wr = csv.writer(f)

#     for data in datas:
#         wr.writerow(data)