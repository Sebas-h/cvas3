
import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('library1.jpg', 0)
imgR = cv2.imread('library2.jpg', 0)

imgL = cv2.imread('house1.jpg', 0)
imgR = cv2.imread('house2.jpg', 0)

stereo = cv2.StereoBM_create(numDisparities=64, blockSize=7)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()