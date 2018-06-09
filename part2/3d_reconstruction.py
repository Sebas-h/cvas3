
from matplotlib import pyplot as plt

import numpy as np
from skimage import io
from skimage.color import rgb2grey
from skimage.draw import ellipse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot_3d_points(X, X_C1, X_C2):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2],
               c='k', depthshade=True, s=2)
    ax.scatter(X_C1[0], X_C1[1], X_C1[2], c='r')
    ax.scatter(X_C2[0], X_C2[1], X_C2[2], c='b')
    dim = 15
    ax.set_xlim(-dim, dim)
    ax.set_ylim(-dim, dim)
    ax.set_zlim(-dim, dim)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(-100, 90)
    plt.show()


def compute_camera_coordinate(C):
    _, _, v_T = np.linalg.svd(C, full_matrices=True)

    # Final solution is the column of v corresponding to the smallest eigenvalue (so, the last one)
    # or last row in v_T (v transpose)
    return v_T[3, :] / v_T[3, 3]


def compute_3d_point(p1, p2, C1, C2):
    A = np.empty((4, 4))

    A[0] = p1[0] * C1[2].T - C1[1].T
    A[1] = p1[1] * C1[2].T - C1[0].T
    A[2] = p2[0] * C2[2].T - C2[1].T
    A[3] = p2[1] * C2[2].T - C2[0].T

    _, _, v_T = np.linalg.svd(A, full_matrices=True)

    # Final solution is the column of v corresponding to the smallest eigenvalue (so, the last one)
    # or last row in v_T (v transpose)
    return v_T[3, :] / v_T[3, 3]


def compute_3d_points(matches, C1, C2):
    # number of corresponding points
    N = matches.shape[0]

    matches_1 = matches[:, [0, 1]]
    matches_2 = matches[:, [2, 3]]

    X = [compute_3d_point(matches_1[i, :], matches_2[i, :], C1, C2)
         for i in range(N)]

    return np.array(X)


if __name__ == "__main__":
    image1 = io.imread('library1.jpg')
    image2 = io.imread('library2.jpg')
    cam1 = np.loadtxt('library1_camera.txt')
    cam2 = np.loadtxt('library2_camera.txt')
    matches = np.loadtxt('library_matches.txt')

    # image1 = io.imread('house1.jpg')
    # image2 = io.imread('house2.jpg')
    # cam1 = np.loadtxt('house1_camera.txt')
    # cam2 = np.loadtxt('house2_camera.txt')
    # matches = np.loadtxt('house_matches.txt')


    X = compute_3d_points(matches, cam1, cam2)

    X_c1 = compute_camera_coordinate(cam1)
    X_c2 = compute_camera_coordinate(cam2)

    plot_3d_points(X, X_c1, X_c2)


