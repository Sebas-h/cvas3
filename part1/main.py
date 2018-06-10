from skimage import io, transform
from image_stitching import *
from ransac import *


def main():
    # img1_name = 'unimaas1/unimaas_a.jpg'
    # img2_name = 'unimaas1/unimaas_b.jpg'
    # img1_name = 'unimaas2/unimaas_a.jpg'
    # img2_name = 'unimaas2/unimaas_b.jpg'
    img1_name = 'building/a.jpg'
    img2_name = 'building/b.jpg'

    print(img1_name)
    image_left = io.imread('part1/data/' + img1_name)
    image_right = io.imread('part1/data/' + img2_name)

    show_stitched_image = False

    patch_size = [9,9]
    k = 200             # top k matches that are chosen

    ransac_iters = 500
    ransac_threshold = 5
    ransac_num_inliers = 3
    stitched_image = im_stitch(image_left, image_right, patch_size, k, ransac_iters, ransac_threshold, ransac_num_inliers)
    if show_stitched_image:
        plot_stitched_img(stitched_image)
        



def im_stitch(image_left, image_right, patch_size, k, ransac_iters, ransac_threshold, ransac_num_inliers):
    # HARRIS
    coords_1, coord_subpix_1 = harris_detector(image_left)
    coords_2, coord_subpix_2 = harris_detector(image_right)
    
    # DESCRIBE KEYPOINTS
    descriptors_1 = patch_descriptor(image_left, coords_1, patch_size)
    descriptors_2 = patch_descriptor(image_right, coords_2, patch_size)
    print('patch_size =', patch_size, ', descriptor size =', patch_size[0] * patch_size[1])
    print('#keypoints img1 =', descriptors_1.shape[0])
    print('#keypoints img2 =', descriptors_2.shape[0])
    print('#potential matches =', descriptors_1.shape[0] * descriptors_2.shape[0])

    # CALCULATE DISTS AND CHOOSE K BEST MATCHES 
    #   the ones with closest dist between descriptors
    if k > np.min([descriptors_1.shape[0], descriptors_2.shape[0]]):
        k = np.min([descriptors_1.shape[0], descriptors_2.shape[0]])
    top_matched_points = choose_matching_points(descriptors_1, coords_1,
    coords_2, descriptors_2, k)
    print(k, 'best matches chosen')

    # add x_max to x' to get proper distances in transformation
    img1_x_max = image_left.shape[1]
    top_matched_points[:,5] += img1_x_max

    # ESTIMATE AFFINE TRANSFORM USING RANSAC
    minrq  = 3
    iters  = ransac_iters
    thresh = ransac_threshold
    mininl = ransac_num_inliers
    tform_mat, inliers = estimate_tranform_ransac(top_matched_points, minrq, iters, thresh, mininl)
    
    print('ransac: minreq, iters, threshold, mininl =', minrq, iters, thresh, mininl)
    print('affine transform matrix:\n',tform_mat)
    print('num inliers =', inliers.shape[0], 'out of', k)

    # CALCULATE ERRORS OF INLIERS
    errors = calculate_errors(inliers, tform_mat)
    print('mean errors = ', np.mean(errors))
    print('std errors =', np.std(errors))
    
    # To plot matching points, reverse the addition of x_max to x'
    inliers[:,2] -= img1_x_max
    sorted_inliers_by_err = inliers[np.argsort(errors)]

    matches_to_show = 10
    if matches_to_show > inliers.shape[0]:
        matches_to_show = inliers.shape[0]

    # plot_matching_points(image_left, image_right, sorted_inliers_by_err, matches_to_show)

    stitched_image = stitch_images(image_left, image_right, tform_mat)
    return stitched_image


if __name__ == "__main__":
    print_to_file = False
    if print_to_file:
        import sys
        import os.path
        orig_stdout = sys.stdout

        i = 0
        while os.path.exists("part1/output/res%s.txt" % i):
            i += 1

        f = open("part1/output/res%s.txt" % i, 'w')
        sys.stdout = f
        
        main()

        sys.stdout = orig_stdout
        f.close()
    else:
        main()