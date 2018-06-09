import numpy as np


def ransac(data, n, k, t, d):
    """ransac implementation
    
    Arguments:
        data {array} -- data to fit model to
        n {int} -- number of datapoints required
        k {int} -- number of ransace iterations
        t {int} -- threshold value to be an inlier
        d {int} -- number of inliers for model to be considered
    
    Returns:
        array -- affine transform matrix
    """

    best_model = 0
    best_counter = 0
    best_models_inliers = 0
    best_error = np.inf
    bm_inlier_dists = []

    for _ in range(k):
        # 1)choose n datapoints, required to fit the model
        random_indices = [np.random.randint(data.shape[0]) for _ in range(n)]
        required_datapoints = data[random_indices,:]


        # 2)compute the (potential) model (affine transform matrix) using the n required datapoints
        #       xT = x' ==> T = x' * inverse(x)
        # method 1:
        # a = np.array([required_datapoints[:,0], required_datapoints[:,1], np.repeat([1], n)])
        # b = np.array([required_datapoints[:,2], required_datapoints[:,3], np.repeat([1], n)])
        # potential_model = np.linalg.lstsq(a,b, rcond=None)[0]
        # potential_model = np.concatenate( (test[:2,:], np.array([[0,0,1]]) ) )
        
        # method 2:  (Gives different matrix than method 1)
        tranform_params = least_squares_for_affine(required_datapoints)[0]
        potential_model = np.array([
            [tranform_params[0][0], tranform_params[1][0], tranform_params[2][0]],
            [tranform_params[3][0], tranform_params[4][0], tranform_params[5][0]],
            [0,0,1]
        ])


        # 3) use computed (potential) model to calculate img2 coordinate for potential matching point
        #       and compare with actual value (x',y') of matching point img2
        #       if the difference is within a threshold t
        #       then increase count of good transforms that the potential model gave (called inliers)
        counter_within_t = 0
        inliers = required_datapoints
        other_datapoints = np.delete(data, random_indices, axis=0)
        inliers_dists = []

        for datapoint in other_datapoints:
            # for each datapoint outside the required ones
            #       compute error of datapoint to model
            #       if error < t: add to inliers
            predicted_transformed_point = np.dot(potential_model, np.concatenate((datapoint[:2],[1])))
            # (euclidean) distance from model prediction to actual point:
            distance_pu_to_u = np.linalg.norm(predicted_transformed_point[:2] - datapoint[2:])
            if distance_pu_to_u < t:
                inliers_dists.append(distance_pu_to_u)
                inliers = np.concatenate((inliers,[datapoint]))
                counter_within_t += 1

        # if enough inliers within t, recompute transformation matrix using all inliers
        #       if new model better performance, take this one
        if counter_within_t > best_counter:
            # refit model to all inliers:
            better_model = least_squares_for_affine(inliers)[0]
            best_counter = counter_within_t
            best_model = better_model
            best_models_inliers = inliers
            bm_inlier_dists = inliers_dists

    return best_model, best_models_inliers, best_counter, best_error, bm_inlier_dists


def least_squares_for_affine(inliers):
    # matrix like on slide 35 of Hough-Ransac Lecture pdf
    a = np.array([
        [   np.sum(inliers[:,0] ** 2), 
            np.sum(inliers[:,0]*inliers[:,1]),
            np.sum(inliers[:,0]),
            0,0,0
        ],
        [
            np.sum(inliers[:,0]*inliers[:,1]),
            np.sum(inliers[:,1] ** 2),
            np.sum(inliers[:,1]),
            0,0,0
        ],
        [
            np.sum(inliers[:,0]),
            np.sum(inliers[:,1]),
            inliers.shape[0],
            0,0,0
        ],
        [   
            0,0,0,
            np.sum(inliers[:,0] ** 2), 
            np.sum(inliers[:,0]*inliers[:,1]),
            np.sum(inliers[:,0])
        ],
        [
            0,0,0,
            np.sum(inliers[:,0]*inliers[:,1]),
            np.sum(inliers[:,1] ** 2),
            np.sum(inliers[:,1])
        ],
        [
            0,0,0,
            np.sum(inliers[:,0]),
            np.sum(inliers[:,1]),
            inliers.shape[0]
        ]
    ])
    b = np.array([
        [np.sum(inliers[:,2] * inliers[:, 0])],
        [np.sum(inliers[:,2] * inliers[:,1])],
        [np.sum(inliers[:,2])],
        [np.sum(inliers[:,3] * inliers[:, 0])],
        [np.sum(inliers[:,3] * inliers[:,1])],
        [np.sum(inliers[:,3])]
    ])
    return np.linalg.lstsq(a, b, rcond=None)
