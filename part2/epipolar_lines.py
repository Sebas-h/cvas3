from matplotlib import pyplot as plt

import numpy as np
from skimage import io
from skimage.color import rgb2grey
from skimage.draw import ellipse
import math

plt.style.use('seaborn-whitegrid')

def plot_match_lines(image1, image2, matches):

    N = matches.shape[0]

    plt.figure(1, figsize=(8, 3))
    ax = plt.subplot(111)
    ax.imshow(np.concatenate((image1, image2), axis=1), interpolation='nearest')
    ax.axis((0, image1.shape[1] * 2, image1.shape[0], 0))
    for i in range(N):
        ax.plot([matches[i, 0], matches[i, 2] + image1.shape[1]], matches[i, [1, 3]])

    plt.show()


def plot_match_points(image1, image2, matches):
    plt.figure(1, figsize=(8, 3))
    ax1 = plt.subplot(121)
    ax1.imshow(image1, interpolation='nearest')
    ax1.plot(matches[:, 0], matches[:, 1], 'xr', markersize=5)
    ax1.plot(matches[:, 2], matches[:, 3], 'xb', markersize=5)
    ax1.axis((0, image1.shape[1], image1.shape[0], 0))

    ax2 = plt.subplot(122)
    ax2.imshow(image2, interpolation='nearest')
    ax2.axis((0, image2.shape[1], image2.shape[0], 0))
    ax2.plot(matches[:, 2], matches[:, 3], 'xr', markersize=5)
    ax2.plot(matches[:, 0], matches[:, 1], 'xb', markersize=5)
    plt.show()


def plot_epipolar_lines(matches, image1, image2, F):

    plot_every_ith_line = 10

    image1_w = image1.shape[1]
    image1_h = image1.shape[0]
    image2_w = image2.shape[1]
    image2_h = image2.shape[0]

    N = matches.shape[0]

    matches_1 = np.concatenate((matches[:, [0, 1]], np.ones((N, 1))), 1)
    matches_2 = np.concatenate((matches[:, [2, 3]], np.ones((N, 1))), 1)

    l1 = (F.T.dot(matches_2.T)).T
    l2 = (F.dot(matches_1.T)).T

    plt.figure(1, figsize=(8, 3))
    ax1 = plt.subplot(121)
    ax1.imshow(image1, interpolation='nearest')
    plt.ylim((image1_h, 0))

    ax2 = plt.subplot(122)
    ax2.imshow(image2, interpolation='nearest')
    plt.ylim((image2_h, 0))

    for i, m in enumerate(matches):
        ax1.plot(m[0], m[1], marker='x', markersize=10, color="blue")
        ax2.plot(m[2], m[3], marker='x', markersize=10, color="blue")

    for i, l in enumerate(l1):
        x = np.array([0, int(image1_w)])
        a, b, c = l
        y = -a / b * x - c / b
        if i % plot_every_ith_line == 0:
            ax1.plot(x, y, lw=1, color="red")

    for i, l in enumerate(l2):
        x = np.array([0, int(image2_w)])
        a, b, c = l
        y = -a / b * x - c / b
        if i % plot_every_ith_line == 0:
            ax2.plot(x, y, lw=1, color="red")

    plt.show()


def normalize_points(points):
    centroid = np.mean(points, axis=0)

    rms = math.sqrt(np.sum((points - centroid) ** 2) / points.shape[0])

    norm_factor = math.sqrt(2) / rms

    matrix = np.array([[norm_factor, 0, -norm_factor * centroid[0]],
                       [0, norm_factor, -norm_factor * centroid[1]],
                       [0, 0, 1]])

    pointsh = np.row_stack([points.T, np.ones((points.shape[0]),)])

    new_pointsh = (matrix @ pointsh).T

    new_points = new_pointsh[:, :2]
    new_points[:, 0] /= new_pointsh[:, 2]
    new_points[:, 1] /= new_pointsh[:, 2]

    return matrix, new_points



def estimage_fund_matrix(matches, normalization=True):
    # number of corresponding points
    N = matches.shape[0]

    matches_1 = matches[:, [0, 1]]
    matches_2 = matches[:, [2, 3]]

    # normalization ...
    if normalization:
        mat1, matches_1 = normalize_points(matches_1)
        mat2, matches_2 = normalize_points(matches_2)

    c_0 = matches_1[:, 0] * matches_2[:, 0]
    c_1 = matches_1[:, 1] * matches_2[:, 0]
    c_2 = matches_2[:, 0]
    c_3 = matches_1[:, 0] * matches_2[:, 1]
    c_4 = matches_1[:, 1] * matches_2[:, 1]
    c_5 = matches_2[:, 1]
    c_6 = matches_1[:, 0]
    c_7 = matches_1[:, 1]
    c_8 = np.ones(N)

    W = np.column_stack((c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8))

    _, _, V_t = np.linalg.svd(W, full_matrices=True)

    zero_indecies = np.where(W == 0)
    if zero_indecies[0].shape[0] > 0:
        print('warning: zero in V_t matrix. Linear dependency detected')

    # take last column of V_t as values for fundamental matrix
    F_intermediate = V_t[-1].reshape((3, 3))

    U, S, V_t = np.linalg.svd(F_intermediate, full_matrices=True)

    S_zero = np.zeros((3, 3))
    S_zero[0, 0] = S[0]
    S_zero[1, 1] = S[1]

    F = U.dot(S_zero).dot(V_t)

    if normalization:
        # convert coordinates back
        F = mat2.T.dot(F.dot(mat1))

    F_norm = F / F[2, 2]

    return F_norm


def dists_to_epipol_lines(matches, F):
    # number of correspondences
    N = matches.shape[0]

    matches_1 = np.concatenate((matches[:, [0, 1]], np.ones((N, 1))), 1)
    matches_2 = np.concatenate((matches[:, [2, 3]], np.ones((N, 1))), 1)

    l1 = (F.T.dot(matches_2.T)).T
    l2 = (F.dot(matches_1.T)).T

    sum_dist_1 = 0
    sum_dist_2 = 0

    for i in range(len(l1)):
        a, b, c = l1[i]
        x = matches_1[i, 0]
        y = matches_1[i, 1]
        sum_dist_1 += np.absolute(a*x + b*y + c) / np.sqrt(a**2 + b**2)

    for i in range(len(l2)):
        a, b, c = l2[i]
        x = matches_2[i, 0]
        y = matches_2[i, 1]
        sum_dist_2 += np.absolute(a*x + b*y + c) / np.sqrt(a**2 + b**2)

    avg_dist_1 = sum_dist_1 / len(l1)
    avg_dist_2 = sum_dist_2 / len(l2)

    return avg_dist_1, avg_dist_2


if __name__ == "__main__":
    image1 = io.imread('house1.jpg')
    image2 = io.imread('house2.jpg')
    matches = np.loadtxt('house_matches.txt')

    image1 = io.imread('library1.jpg')
    image2 = io.imread('library2.jpg')
    matches = np.loadtxt('library_matches.txt')

    # plot_match_lines(image1, image2, matches)

    # plot_match_points(image1, image2, matches)

    # estimate fundamental matrix from matches
    F = estimage_fund_matrix(matches)

    avg_dists = dists_to_epipol_lines(matches, F)

    print(avg_dists)

    plot_epipolar_lines(matches, image1, image2, F)









