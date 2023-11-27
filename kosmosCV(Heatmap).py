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

MAX_IMAGES = 600
rawImages = []
for i in range(MAX_IMAGES):
    rawImages.append(np.zeros((960, 1280), np.int16))

for nowFrame in tqdm(range(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)):
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
    
    # Make Heatmap
    rawImages[nowFrame%MAX_IMAGES] = np.array( cv2.GaussianBlur(fgmask, (0, 0), 5), dtype=np.uint64 )

    # cv2.imshow('HEATMAP', heatmap)
    # cv2.waitKey(1)

    if nowFrame%599 == 0:
        grayHeatmap = np.zeros((960, 1280), np.uint64)
        for image in rawImages:
            grayHeatmap = grayHeatmap + image
        
        grayHeatmap = np.array(grayHeatmap, dtype=np.float64)
        grayHeatmap = grayHeatmap*255 /np.max(grayHeatmap)
        grayHeatmap = np.array(grayHeatmap, dtype=np.uint8)
        cv2.imwrite('SectionChangeRate.png', grayHeatmap)

    success, image = vidcap.read()

    if not success:
        break

# cv2.destroyAllWindows()