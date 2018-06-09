import math
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.color import rgb2grey
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.draw import ellipse
from scipy.spatial import distance
from ransac import ransac
from skimage import transform


def compute_patches(image_gray, keypoints, patch_size=(3, 3)):
    patches = []
    img_x_max = image_gray.shape[0]
    img_y_max = image_gray.shape[1]
    for keypoint in keypoints:
        x1 = keypoint[0] - math.floor(patch_size[0] / 2)
        if x1 < 0:
            x1 = 0
        x2 = keypoint[0] + math.floor(patch_size[0] / 2) + 1
        if x2 >= img_x_max:
            x2 = img_x_max
        y1 = keypoint[1] - math.floor(patch_size[1] / 2)
        if y1 < 0:
            y1 = 0
        y2 = keypoint[1] + math.floor(patch_size[1] / 2) + 1
        if y2 >= img_y_max:
            y2 = img_y_max
        patch = image_gray[x1:x2, y1:y2]
        patch = patch.flatten()
        patches.append(patch)
    return np.array(patches)


image1 = io.imread('part1/data/unimaas1/unimaas_a.jpg')
image2 = io.imread('part1/data/unimaas1/unimaas_b.jpg')

image1 = rgb2grey(image1)
image1 = transform.rotate(image1, 5)
image2 = rgb2grey(image2)
print('img1 gray shape', image1.shape)

test = corner_harris(image1)

coords_img1 = corner_peaks(corner_harris(image1), min_distance=5)
coords_subpix_img1 = corner_subpix(image1, coords_img1, window_size=20)
print('corner peaks img1', coords_img1.shape)
print('corner subpix', coords_subpix_img1.shape)

coords_img2 = corner_peaks(corner_harris(image2), min_distance=5)
coords_subpix_img2 = corner_subpix(image2, coords_img2, window_size=20)

# -----

# PATCHES
patches_img1 = compute_patches(image1, coords_img1)
print('patches1', patches_img1.shape)
patches_img2 = compute_patches(image2, coords_img2)


# NORMALIZE
patches_img1 = (patches_img1 - np.min(patches_img1)) / \
    (np.max(patches_img1) - np.min(patches_img1))
patches_img2 = (patches_img2 - np.min(patches_img2)) / \
    (np.max(patches_img2) - np.min(patches_img2))


# DISTANCES
distances = distance.cdist(patches_img1, patches_img2)
print('distances:', distances.shape)


# BEST K MATCHED POINTS
k = 500
t = []
for i in range(distances.shape[0]):
    for j in range(distances.shape[1]):
        # (dist_row, dist_col, dist_value, x, y, x', y')
        t.append((i, j, distances[i, j],
                  coords_img1[i][0], coords_img1[i][1],
                  coords_img2[j][0], coords_img2[j][1])
                 )
t = np.array(t)
ordered_dists = t[t[:, 2].argsort()]
if ordered_dists.shape[0] < k:
    k = ordered_dists.shape[0]
top_k_dists = ordered_dists[:k, :]


# RANSAC TO ESTIMATE AFFINE TRANSFORM
data = top_k_dists[:, 3:]
model = 0  # not sure about this parameter
n = 3
num_iters = 100
threshold = 20  # distance threshold to model to be counted as inlier
d = 20
model, inliers, count, error, dists = ransac(data, model, n, num_iters, threshold, d)
print('#inliers', inliers.shape[0])
print('#dists:', len(dists))
print('mean dists:', np.mean(dists))
print('stdev dists:', np.std(dists))


# STICH IMAGES
affine_transform_matrix = np.array([
    [model[0][0], model[1][0], model[2][0]],
    [model[3][0], model[4][0], model[5][0]],
    [0,0,1]
])
tform = transform.AffineTransform(matrix=affine_transform_matrix)
img1_transformed = transform.warp(image1, tform)



plt.figure(1)
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)
ax1.imshow(image1, cmap='gray')
ax2.imshow(img1_transformed, cmap='gray')
ax3.imshow(image2,cmap='gray')
plt.show()

# plt.figure(1, figsize=(8, 3))
# ax1 = plt.subplot(121)
# ax1.imshow(image1, interpolation='nearest', cmap='gray')
# ax1.plot(coords_img1[:, 1], coords_img1[:, 0], '.b', markersize=3)
# ax1.plot(coords_subpix_img1[:, 1],
#          coords_subpix_img1[:, 0], '+r', markersize=15)
# ax1.axis((0, 2000, 2000, 0))

# ax2 = plt.subplot(122)
# ax2.imshow(image2, interpolation='nearest', cmap='gray')
# ax2.plot(coords_img2[:, 1], coords_img2[:, 0], '.b', markersize=3)
# ax2.plot(coords_subpix_img2[:, 1],
#          coords_subpix_img2[:, 0], '+r', markersize=15)
# ax2.axis((0, 2000, 2000, 0))


# plt.figure(2, figsize=(8, 3))
# ax3 = plt.subplot(111)
# ax3.imshow(image1, interpolation='nearest', cmap='gray')
# ax3.plot(coords_img1[:, 1], coords_img1[:, 0], '.b', markersize=3)
# # ax3.imshow(np.concatenate((image1, image2), axis=1), cmap='gray')

# plt.show()
