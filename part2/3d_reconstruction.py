
from matplotlib import pyplot as plt

import numpy as np
from skimage import io
from skimage.color import rgb2grey
from skimage.draw import ellipse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


plt.style.use('seaborn-whitegrid')


def plot_match_points(image1, image2, matches, color_map):
    """
    plots points of the matched points of the other image in each image
        :param image1: 
        :param image2: 
        :param matches: 
        :param color_map:
    """

    # create figure
    fig = plt.figure(1, figsize=(8, 3))

    # plot image1
    ax1 = plt.subplot(121)
    ax1.imshow(image1, interpolation='nearest')
    ax1.axis((0, image1.shape[1], image1.shape[0], 0))

    for i, _ in enumerate(matches):
        color = color_map[i]
        ax1.plot(matches[i, 0], matches[i, 1], color=color, marker='x', markersize=10)

    # plot image 2
    ax2 = plt.subplot(122)
    ax2.imshow(image2, interpolation='nearest')
    ax2.axis((0, image2.shape[1], image2.shape[0], 0))

    for i, _ in enumerate(matches):
        color = color_map[i]
        ax2.plot(matches[i, 2], matches[i, 3], color=color, marker='x', markersize=10)

    plt.show()

    # fig.savefig("matched_lines_colored_library.pdf", bbox_inches='tight')

def plot_3d_points(X, X_C1, X_C2, color_map):
    """
    docstring here
        :param X: 
        :param X_C1: 
        :param X_C2: 
        :param color_map:
    """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')

    # plot the points
    for i, x in enumerate(X):
        ax.scatter(x[0], x[1], x[2], c=color_map[i], depthshade=True, s=2)
    
    # plot camera positions
    ax.scatter(X_C1[0], X_C1[1], X_C1[2], c='r')
    ax.scatter(X_C2[0], X_C2[1], X_C2[2], c='b')

    # set dimensions and axis lables
    dim = 15
    ax.set_xlim(-dim, dim)
    ax.set_ylim(-dim, dim)
    ax.set_zlim(-dim, dim)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    # set angles
    elev = 30
    azim = -90
    ax.view_init(elev, azim)

    plt.show()

    fig.savefig("3d_library_angle2.pdf", bbox_inches='tight')


def compute_camera_coordinate(C):
    """
    docstring here
        :param C: 
    """

    _, _, v_T = np.linalg.svd(C, full_matrices=True)

    # Final solution is the column of v corresponding to the smallest eigenvalue (so, the last one)
    # or last row in v_T (v transpose)
    return v_T[3, :] / v_T[3, 3]


def compute_3d_point(p1, p2, C1, C2):
    """
    docstring here
        :param p1: 
        :param p2: 
        :param C1: 
        :param C2: 
    """

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
    """
    docstring here
        :param matches: 
        :param C1: 
        :param C2: 
    """

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

    y_max = np.ceil(np.max([np.max(matches[:, 1])]))
    y_min = np.ceil(np.min([np.min(matches[:, 1])]))

    print(y_max)
    print(y_min)

    # create a color map
    diff = y_max - y_min
    step_size = np.ceil(diff / 10)

    c = np.ceil((matches[:, 1] - y_min) / step_size).astype(int)

    c_map = ['red', 'orange', 'gold', 'lime', 'darkgreen',
             'darkcyan', 'cyan', 'blue', 'navy', 'darkviolet', 'magenta']
    
    color_map = [c_map[i] for _, i in enumerate(c)]

    # plot_match_points(image1, image2, matches, color_map)

    X = compute_3d_points(matches, cam1, cam2)

    X_c1 = compute_camera_coordinate(cam1)
    X_c2 = compute_camera_coordinate(cam2)

    print(X_c1)
    print(X_c2)

    plot_3d_points(X, X_c1, X_c2, color_map)
