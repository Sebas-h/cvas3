import math
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.color import rgb2grey
from skimage.feature import corner_harris, corner_subpix, corner_peaks, plot_matches
from skimage.draw import ellipse
from scipy.spatial import distance
from ransac import ransac
from skimage import transform
from skimage import measure


def compute_patches(image_gray, keypoints, patch_size=(3, 3)):
    patches = []
    change_x = math.floor(patch_size[0] / 2)
    change_y = math.floor(patch_size[1] / 2)
    for keypoint in keypoints:
        x1 = keypoint[0] - change_x
        x2 = keypoint[0] + change_x + 1
        y1 = keypoint[1] - change_y
        y2 = keypoint[1] + change_y + 1

        patch = image_gray[x1:x2, y1:y2]
        patch = patch.flatten()
        patches.append(patch)
    return np.array(patches)


# LOAD IMAGES
image1 = io.imread('part1/data/unimaas1/unimaas_a.jpg')
image2 = io.imread('part1/data/unimaas1/unimaas_b.jpg')

image1 = rgb2grey(image1)
image2 = rgb2grey(image2)

# HARRIS CORNER DETECTION
coords_img1 = corner_peaks(corner_harris(image1), min_distance=5)
coords_subpix_img1 = corner_subpix(image1, coords_img1, window_size=20)

coords_img2 = corner_peaks(corner_harris(image2), min_distance=5)
coords_subpix_img2 = corner_subpix(image2, coords_img2, window_size=20)

# ---------------------------------------------------
# ---------------------------------------------------

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
k = 200
t = []
for i in range(distances.shape[0]):
    for j in range(distances.shape[1]):
        # (dist_row, dist_col, dist_value, x, y, x', y')
        t.append((i, j, distances[i, j],
                  coords_img1[i][1], coords_img1[i][0],
                  coords_img2[j][1], coords_img2[j][0])
                 )
t = np.array(t)
ordered_dists = t[t[:, 2].argsort()]
if ordered_dists.shape[0] < k:
    k = ordered_dists.shape[0]
top_k_dists = ordered_dists[:k, :]


# RANSAC TO ESTIMATE AFFINE TRANSFORM MATRIX
data = top_k_dists[:, 3:]
min_num_datapoints_to_fit_model = 3
num_iters = 100
threshold = 2  # distance threshold to model to be counted as inlier
min_num_inliers = 5
model, inliers, count, error, dists = ransac(data, 
    min_num_datapoints_to_fit_model, num_iters, threshold, min_num_inliers)
print('#inliers', inliers.shape[0])
print('#dists:', len(dists))
print('mean dists:', np.mean(dists))
print('stdev dists:', np.std(dists))


# TRANFORM IMAGE
affine_transform_matrix = np.array([
    [model[0][0], model[1][0], model[2][0]],
    [model[3][0], model[4][0], model[5][0]],
    [0,0,1]
])
tform = transform.AffineTransform(matrix=affine_transform_matrix)
print('my ransac tform:\n', affine_transform_matrix)
img1_transformed = transform.warp(image1, tform)



# -------------------------------------------------
# TEST/VALIDATE
# -------------------------------------------------
src = top_k_dists[:,3:5]
dst = top_k_dists[:,5:7]
# tform3 = transform.AffineTransform()
# tform3.estimate(src, dst)
# print('estimated tform3:\n', tform3.params)

model_robust, inliers = measure.ransac((src, dst), transform.AffineTransform, min_samples=3,
                               residual_threshold=2, max_trials=100)
print('skimg ransac model:\n', model_robust.params)
print('skimg ransac #inliers', inliers.shape)
warped = transform.warp(image1, model_robust)
# -------------------------------------------------|



# -------------------------------------------------
#               PLOTTING (+ STITCHING)
# -------------------------------------------------

# PLOT TRANFORMATION OF IMG1
# plt.figure(1)
# ax1 = plt.subplot(131)
# ax2 = plt.subplot(132)
# ax3 = plt.subplot(133)
# ax1.imshow(image1, cmap='gray')
# ax2.imshow(img1_transformed, cmap='gray')
# ax3.imshow(image2, cmap='gray')


# PLOT MATCHING POINTS:
# plt.figure(2)
# ax4 = plt.subplot(111)
# src = np.column_stack((src[:,1], src[:,0]))
# dst = np.column_stack((dst[:,1], dst[:,0]))
# src = src[:10]
# dst = dst[:10]
# inlier_idxs = np.nonzero(src)[0]
# mm = np.column_stack((inlier_idxs, inlier_idxs))
# plot_matches(ax4, image1, image2, src, dst, mm)


# PLOT MERGED IMAGES:
(h0, w0) = image1.shape
(h1, w1) = warped.shape
(h2, w2) = image2.shape
adjacent = np.zeros((max(h0, h2), w0 + w2))
adjacent[0:h0, 0:w0] = image1
adjacent[0:h2, w0:w0+w2] = image2
merge = np.zeros((max(h1, h2), w1 + w2))
merge[0:h1, 0:w1] = warped
merge[0:h2, w1:w1+w2] = image2
plt.figure(3)
# ax5 = plt.subplot(211)
ax6 = plt.subplot(111)
# ax5.imshow(adjacent, cmap='gray')
ax6.imshow(merge, cmap='gray')

plt.show()

# PLOT HARRIS CORNERS
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
# plt.show()
