import numpy as np

# Source: wikipedia 'RANSAC'
# Given:
#     data – a set of observed data points
#     model – a model that can be fitted to data points
#     n – minimum number of data points required to fit the model
#     k – maximum number of iterations allowed in the algorithm
#     t – threshold value to determine when a data point fits a model
#     d – number of close data points required to assert that a model fits well to data

# Return:
#     bestfit – model parameters which best fit the data (or nul if no good model is found)

# iterations = 0
# bestfit = nul
# besterr = something really large
# while iterations < k {
#     maybeinliers = n randomly selected values from data
#     maybemodel = model parameters fitted to maybeinliers
#     alsoinliers = empty set
#     for every point in data not in maybeinliers {
#         if point fits maybemodel with an error smaller than t
#              add point to alsoinliers
#     }
#     if the number of elements in alsoinliers is > d {
#         % this implies that we may have found a good model
#         % now test how good it is
#         bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
#         thiserr = a measure of how well model fits these points
#         if thiserr < besterr {
#             bestfit = bettermodel
#             besterr = thiserr
#         }
#     }
#     increment iterations
# }
# return bestfit



def ransac(data, model, n, k, t, d):
    """
    data – a set of observed data points
    model – a model that can be fitted to data points
    n – minimum number of data points required to fit the model
    k – maximum number of iterations allowed in the algorithm
    t – threshold value to determine when a data point fits a model
    d – number of close data points required to assert that a model fits well to data
    """
    best_model = 0
    best_counter = 0
    best_models_inliers = 0
    best_error = np.inf

    for _ in range(k):
        # 1)choose n datapoints, required to fit the model
        random_indices = [np.random.randint(data.shape[0]) for _ in range(n)]
        required_datapoints = data[random_indices,:]

        # 2)compute the (potential) model (affine transform matrix) using the n required datapoints
        #       aT=b .. T = b * a^-1
        a = np.array([required_datapoints[:,0], required_datapoints[:,1], np.repeat([1], n)])
        if np.linalg.det(a) == 0:
            a_inv = np.linalg.pinv(a)
        else:
            a_inv = np.linalg.inv(a)
        b = np.array([required_datapoints[:,2], required_datapoints[:,3], np.repeat([1], n)])
        potential_model = np.dot(b, a_inv)
        potential_model = potential_model[:2,:]

        # 3) use computed (potential) model to calculate img2 coordinate for potential matching point
        #       and compare with actual value (x',y') of matching point img2
        #       if the difference is within a threshold t
        #       then increase count of good transforms that the potential model gave (called inliers)
        counter_within_t = 0
        inliers = required_datapoints
        other_datapoints = np.delete(data, random_indices, axis=0)

        for datapoint in other_datapoints:
            # for each datapoint outside the required ones
            #       compute error of datapoint to model
            #       if error < t: add to inliers
            predicted_transformed_point = np.dot(potential_model, np.concatenate((datapoint[:2],[1])))
            # (euclidean) distance from model prediction to actual point:
            if np.linalg.norm(predicted_transformed_point - datapoint[2:]) < t:
                inliers = np.concatenate((inliers,[datapoint]))
                counter_within_t += 1

        # if enough inliers within t, recompute transformation matrix using all inliers
        #       if new model better performance, take this one
        # [not sure this is correct, the assignment wants ransac to return biggest set of inliers and not a model, :/]
        if counter_within_t > d:
            aa = np.array([inliers[:,0], inliers[:,1], np.repeat([1], inliers.shape[0])])
            bb = np.array([inliers[:,2], inliers[:,3], np.repeat([1], inliers.shape[0])])
            better_model = np.dot(bb, np.linalg.pinv(aa))[:2,:]
            sum_error = 0
            for inlier in inliers:
                predicted_transformed_point = np.dot(better_model, np.concatenate((inlier[:2],[1])))
                sum_error += np.linalg.norm(predicted_transformed_point - inlier[2:])
            avg_error = sum_error / inliers.shape[0]
            if avg_error < best_error:
                best_error = avg_error
                best_model = better_model
                best_counter = counter_within_t
                best_models_inliers = inliers

    # remember the potential model and how many inliers it gave
    # do steps 1, 2, 3 for k times (k = parameter of number of iterations of alg)
    # choose the best model and return this

    return best_model, best_models_inliers, best_counter, best_error
