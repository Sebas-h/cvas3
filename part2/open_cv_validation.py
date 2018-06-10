import cv2
import numpy as np
from skimage import io
from skimage.color import rgb2grey
from matplotlib import pyplot as plt


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1.astype(int)), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2.astype(int)), 5, color, -1)
    return img1, img2


imagePath1 = 'house1.jpg' 
imagePath2 = 'house2.jpg'
matchesFile = 'house_matches.txt'

# imagePath1 = 'library1.jpg'
# imagePath2 = 'library2.jpg'
# matchesFile = 'library_matches.txt'

img1 = cv2.imread(imagePath1, 0)  # queryimage # left image
img2 = cv2.imread(imagePath2, 0) # trainimage # right image
matches = np.loadtxt(matchesFile)

pts1 = matches[:, [0, 1]]
pts2 = matches[:, [2, 3]]

pts1 = np.int32(matches[:, [0, 1]])
pts2 = np.int32(matches[:, [2, 3]])


F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

print(F)

fig = plt.figure(1)

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.show()

fig.savefig("epipolar_lines_house_opencv.pdf", bbox_inches='tight')