import cv2
import numpy as np

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


while success:
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

    cv2.imshow('ORIGINAL', image)
    cv2.imshow('MASK', fgmask)

    # gray_float = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.float64)
    # gray_float = gray_float*255 / np.max(gray_float)
    # gray = np.array(gray_float, dtype=np.uint8)

    # cv2.imwrite("Brightness.png", gray)
    # exit()


    cv2.waitKey(5)

    success, image = vidcap.read()

cv2.waitKey(0)
cv2.destroyAllWindows()