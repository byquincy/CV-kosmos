import cv2
import numpy as np
import csv
from tqdm import tqdm


# FILE = 'c-elegance_5x.mp4'
FILE = 'second_5x.mp4'
# FILE = 'first_5x.mp4'

fgbg = cv2.createBackgroundSubtractorMOG2(
    varThreshold=10,
    detectShadows=False
)

vidcap = cv2.VideoCapture(FILE)
success,image = vidcap.read()

count = 0

boxCounts = []
for temp in tqdm(range(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)):
    fgmask = fgbg.apply(image)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)

    boxCounts.append(0)

    for index, centroid in enumerate(centroids):
        if stats[index][0] == 0 and stats[index][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue


        x, y, width, height, area = stats[index]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > 100:
            boxCounts[-1]+= 1
            cv2.circle(image, (centerX, centerY), 1, (0, 255, 0), 2)
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255))

    success, image = vidcap.read()

    if not success:
        break

with open('boxCounts.csv', 'w') as f:
    wr = csv.writer(f)

    for data in boxCounts:
        wr.writerow((str(data), ))