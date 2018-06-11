import math, os
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.color import rgb2grey
from skimage.feature import corner_harris, corner_subpix, corner_peaks, plot_matches
from skimage.draw import ellipse
from scipy.spatial import distance
from ransac import ransac, calculate_errors
from skimage import transform
from skimage import measure
import cv2


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


def harris_detector(image):
    image = rgb2grey(image)
    coords_img = corner_peaks(corner_harris(image), min_distance=5)
    coords_subpix_img = corner_subpix(image, coords_img, window_size=20)
    return coords_img, coords_subpix_img
    

def patch_descriptor(image, coords, patch_size):
    patches_img = compute_patches(image, coords, patch_size=patch_size)
    # normalize patches:
    patches_img = (patches_img - np.min(patches_img)) / \
        (np.max(patches_img) - np.min(patches_img))
    return patches_img


def choose_matching_points(descriptors_1, coords_1,
    coords_2, descriptors_2, k, distances_measure='euclidean'):
    distances = distance.cdist(descriptors_1, descriptors_2, metric=distances_measure)
    distances_sort = np.argmin(distances, axis=1)

    matched_points = []

    for idx, val in enumerate(distances_sort):
        matched_points.append((
            idx, val, distances[idx][val],
            coords_1[idx][1],
            coords_1[idx][0],
            coords_2[val][1], 
            coords_2[val][0]
        ))

    print(distances_sort.shape)
    print(len(matched_points))

    # for i in range(distances.shape[0]):
    #     for j in range(distances.shape[1]):
    #         # (dist_row, dist_col, dist_value, x, y, x', y')
    #         matched_points.append(
    #             (
    #                 i, j, distances[i, j],
    #                 coords_1[i][1],
    #                 coords_1[i][0],
    #                 coords_2[j][1], 
    #                 coords_2[j][0]
    #             )
    #         )

    matched_points = np.array(matched_points)
    ordered_dists = matched_points[matched_points[:, 2].argsort()]
    if ordered_dists.shape[0] < k:
        k = ordered_dists.shape[0]
    return ordered_dists[:k, :]


def estimate_tranform_ransac(top_matched_points, ransac_min_required_points, ransac_iters, 
    ransac_threshold, ransac_min_inliers):
    data = top_matched_points[:, 3:]
    model, inliers, _, _, _ = ransac(data, 
        ransac_min_required_points, ransac_iters, ransac_threshold, ransac_min_inliers)
    affine_transform_matrix = np.array([
        [model[0][0], model[1][0], model[2][0]],
        [model[3][0], model[4][0], model[5][0]],
        [0,0,1]
    ])
    return affine_transform_matrix, inliers


# -------------------------------------------------
# TEST/VALIDATE
# -------------------------------------------------
def skimg_ransac_validate(top_matched_points):
    src_skr = top_matched_points[:,3:5]
    dst_skr = top_matched_points[:,5:7]
    model_robust, skimg_inliers = measure.ransac((src_skr, dst_skr), 
                                    transform.AffineTransform, min_samples=3,
                                    residual_threshold=2, max_trials=100)
    print('skimg ransac model:\n', model_robust.params)
    print('skimg ransac #inliers', skimg_inliers.shape)
    return model_robust, skimg_inliers


# -------------------------------------------------
# PLOTTING
# -------------------------------------------------

def plot_transformation(image1, image2, affine_transform_matrix):
    plt.figure(1, dpi=200)
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    ax1.imshow(image1, cmap='gray')
    tform = transform.AffineTransform(matrix=affine_transform_matrix)
    img1_transformed = transform.warp(image1, tform.inverse)
    ax2.imshow(img1_transformed, cmap='gray')
    ax3.imshow(image2, cmap='gray')
    plt.show()

def plot_matching_points(image1, image2, sorted_inliers_by_err, num_matches_shown):
    fig2 = plt.figure(2, dpi=200)
    ax4 = plt.subplot(111)
    src = sorted_inliers_by_err[:,:2]
    dst = sorted_inliers_by_err[:,2:]
    src = np.column_stack((src[:,1], src[:,0]))
    dst = np.column_stack((dst[:,1], dst[:,0]))
    src = src[:num_matches_shown]
    dst = dst[:num_matches_shown]
    inlier_idxs = np.nonzero(src)[0]
    matches = np.column_stack((inlier_idxs, inlier_idxs))
    plot_matches(ax4, image1, image2, src, dst, matches)
    i = 0
    while os.path.exists("part1/results/res%s.png" % i):
        i += 1
    resn = "part1/results/res"+str(i)+".png"
    fig2.savefig(resn, dpi=200) 
    plt.show()


def stitch_images(image1, image2, affine_transform_matrix):
    image1 = rgb2grey(image1)
    image2 = rgb2grey(image2)
    tform = transform.AffineTransform(matrix=affine_transform_matrix)
    img1_transformed = transform.warp(image1, tform.inverse)
    (h1, w1) = img1_transformed.shape
    (h2, w2) = image2.shape
    merge = np.zeros((max(h1, h2), w1 + w2))
    merge[0:h1, 0:w1] = img1_transformed
    merge[0:h2, w1:w1+w2] = image2
    return merge
    
def plot_stitched_img(stitched_img):
    fig = plt.figure(3, dpi=200)
    ax6 = plt.subplot(111)
    ax6.imshow(stitched_img, cmap='gray')
    i = 0
    while os.path.exists("part1/results/stitch%s.png" % i):
        i += 1
    resn = "part1/results/stitch"+str(i)+".png"
    fig.savefig(resn, dpi=200) 
    plt.show()

def plot_harris_corners(image1, image2, 
        coords_img1, coords_subpix_img1,
        coords_img2, coords_subpix_img2):
    plt.figure(1, dpi=200)
    ax1 = plt.subplot(121)
    ax1.imshow(image1, interpolation='nearest', cmap='gray')
    ax1.plot(coords_img1[:, 1], coords_img1[:, 0], '.b', markersize=3)
    ax1.plot(coords_subpix_img1[:, 1],
             coords_subpix_img1[:, 0], '+r', markersize=15)
    ax1.axis((0, 2000, 2000, 0))
    ax2 = plt.subplot(122)
    ax2.imshow(image2, interpolation='nearest', cmap='gray')
    ax2.plot(coords_img2[:, 1], coords_img2[:, 0], '.b', markersize=3)
    ax2.plot(coords_subpix_img2[:, 1],
             coords_subpix_img2[:, 0], '+r', markersize=15)
    ax2.axis((0, 2000, 2000, 0))
    plt.show()
