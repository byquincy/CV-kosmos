import cv2
import numpy as np

img = cv2.imread("BW Sum.png")

print(
    np.sum(img) / (1280*960*255*3)
)