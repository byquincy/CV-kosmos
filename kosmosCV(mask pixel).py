import cv2
import numpy as np
import csv
from tqdm import tqdm

csvLine = []

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


for temp in tqdm(range(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)):
    fgmask = fgbg.apply(image)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)


    for index, centroid in enumerate(centroids):
        if stats[index][0] == 0 and stats[index][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue


        x, y, width, height, area = stats[index]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > 100:
            cv2.circle(image, (centerX, centerY), 1, (0, 255, 0), 2)
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255))

    # cv2.imshow('ORIGINAL', image)
    # cv2.imshow('MASK', fgmask)
    # cv2.waitKey(1)

    csvLine.append(int(np.sum(fgmask)//255))

    success, image = vidcap.read()

    if not success:
        break

# cv2.destroyAllWindows()

with open('maskPixel.csv', 'w') as f:
    wr = csv.writer(f)

    for data in csvLine:
        wr.writerow((str(data), ))