import math

import numpy as np
from matplotlib import pyplot as plt

from skimage import io
from skimage.color import rgb2grey
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.draw import ellipse
from scipy.spatial import distance


def compute_patches(image_gray, keypoints, patch_size=(3,3)):
    patches = []
    img_x_max = image_gray.shape[0]
    img_y_max = image_gray.shape[1]
    for keypoint in keypoints:
        x1 = keypoint[0] - math.floor(patch_size[0] / 2)
        if x1 < 0: x1 = 0
        x2 = keypoint[0] + math.floor(patch_size[0] / 2) + 1
        if x2 >= img_x_max: x2 = img_x_max        
        y1 = keypoint[1] - math.floor(patch_size[1] / 2)
        if y1 < 0: y1 = 0
        y2 = keypoint[1] + math.floor(patch_size[1] / 2) + 1
        if y2 >= img_y_max: y2 = img_y_max
        patches.append(image_gray[x1:x2,y1:y2])        
    return np.array(patches)


image1 = io.imread('part1/data/unimaas1/unimaas_a.jpg')
image2 = io.imread('part1/data/unimaas1/unimaas_b.jpg')

image1 = rgb2grey(image1)
image2 = rgb2grey(image2)
print('img1 gray shape', image1.shape)

test = corner_harris(image1)
print('harris respoinse img1', test.shape)

coords_img1 = corner_peaks(corner_harris(image1), min_distance=5)
coords_subpix_img1 = corner_subpix(image1, coords_img1, window_size=20)
print('corner peaks img1', coords_img1.shape)
print('corner subpix', coords_subpix_img1.shape)

coords_img2 = corner_peaks(corner_harris(image2), min_distance=5)
coords_subpix_img2 = corner_subpix(image2, coords_img2, window_size=20)

# PATCHES
patches_img1 = compute_patches(image1, coords_img1)
descriptors_img1_1d = patches_img1.reshape(-1, patches_img1.shape[-1])
print('patches1',descriptors_img1_1d.shape)
patches_img2 = compute_patches(image2, coords_img2)
descriptors_img2_1d = patches_img2.reshape(-1, patches_img2.shape[-1])

# NORMALIZE
for i in range(descriptors_img1_1d.shape[0]):
    descriptors_img1_1d[i] = (descriptors_img1_1d[i] - np.min(descriptors_img1_1d)) /\
                        (np.max(descriptors_img1_1d) - np.min(descriptors_img1_1d))

for i in range(descriptors_img2_1d.shape[0]):
    descriptors_img2_1d[i] = (descriptors_img2_1d[i] - np.min(descriptors_img2_1d)) /\
                        (np.max(descriptors_img2_1d) - np.min(descriptors_img2_1d))



print(descriptors_img1_1d.shape)
print(descriptors_img2_1d.shape)
distances = distance.cdist(descriptors_img1_1d, descriptors_img2_1d)
print(distances[:5,:5])
print(distances.shape)


# a = divmod(distances.argmin(), distances.shape[1])
# print(a)
# print(distances[687,753])

# test = np.zeros((distances.shape[0] * distances.shape[1]))
t = []
for i in range(distances.shape[0]):
    for j in range(distances.shape[1]):
        t.append((int(i),int(j),distances[i,j]))
# (row,col,dist_value)
t = np.array(t)
# print(t)
print(t.shape)


ordered_dists = t[ t[:,2].argsort() ]

best_k_dists = ordered_dists[:10,:]

print(best_k_dists)









# plt.figure(1, figsize=(8, 3))
# ax1 = plt.subplot(121)
# ax1.imshow(image1, interpolation='nearest', cmap='gray')
# ax1.plot(coords_img1[:, 1], coords_img1[:, 0], '.b', markersize=3)
# ax1.plot(coords_subpix_img1[:, 1], coords_subpix_img1[:, 0], '+r', markersize=15)
# ax1.axis((0, 2000, 2000, 0))

# ax2 = plt.subplot(122)
# ax2.imshow(image2, interpolation='nearest', cmap='gray')
# ax2.plot(coords_img2[:, 1], coords_img2[:, 0], '.b', markersize=3)
# ax2.plot(coords_subpix_img2[:, 1], coords_subpix_img2[:, 0], '+r', markersize=15)
# ax2.axis((0, 2000, 2000, 0))


# plt.figure(2, figsize=(8,3))
# ax3 = plt.subplot(111)
# ax3.imshow(test, cmap='gray')

# plt.show()


