from matplotlib import pyplot as plt

from skimage import io
from skimage.color import rgb2grey
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.draw import ellipse

image1 = io.imread('data/unimaas1/unimaas_a.jpg')
image2 = io.imread('data/unimaas1/unimaas_b.jpg')

image1 = rgb2grey(image1)
image2 = rgb2grey(image2)

coords_img1 = corner_peaks(corner_harris(image1), min_distance=5)
coords_subpix_img1 = corner_subpix(image1, coords_img1, window_size=20)

coords_img2 = corner_peaks(corner_harris(image2), min_distance=5)
coords_subpix_img2 = corner_subpix(image2, coords_img2, window_size=20)

plt.figure(1, figsize=(8, 3))
ax1 = plt.subplot(121)
ax1.imshow(image1, interpolation='nearest', cmap=plt.cm.gray)
ax1.plot(coords_img1[:, 1], coords_img1[:, 0], '.b', markersize=3)
ax1.plot(coords_subpix_img1[:, 1], coords_subpix_img1[:, 0], '+r', markersize=15)
ax1.axis((0, 2000, 2000, 0))

ax2 = plt.subplot(122)
ax2.imshow(image2, interpolation='nearest', cmap=plt.cm.gray)
ax2.plot(coords_img2[:, 1], coords_img2[:, 0], '.b', markersize=3)
ax2.plot(coords_subpix_img2[:, 1], coords_subpix_img2[:, 0], '+r', markersize=15)
ax2.axis((0, 2000, 2000, 0))
plt.show()


