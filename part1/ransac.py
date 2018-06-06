import numpy as np


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

    for _ in range(k):
        # 1)choose n datapoints, required to fit the model
        random_indices = [np.random.randint(data.shape[0]) for _ in range(n)]
        required_datapoints = data[random_indices,:]

        # 2)compute the (potential) model (affine transform matrix) using the n required datapoints
        #       aT=b .. T = b * a^-1
        potential_model = np.dot(
            np.array([required_datapoints[:,2], required_datapoints[:,3], np.repeat([1], n)]),
            np.linalg.inv(np.array([required_datapoints[:,0], required_datapoints[:,1], np.repeat([1], n)]))
        )
        potential_model = potential_model[:2,:]

        # 3) use computed (potential) model to calculate img2 coordinate for potential matching point
        #       and compare with actual value (x',y') of matching point img2
        #       if the difference is within a threshold t
        #       then increase count of good transforms that the potential model gave (called inliers)
        counter_within_t = 0
        other_datapoints = np.delete(data, random_indices, axis=0)
        for datapoint in other_datapoints:
            predicted_transformed_point = np.dot(
                potential_model,
                np.concatenate((datapoint[:2],[1]))
            )
            difference = np.abs(predicted_transformed_point - datapoint[2:])
            if np.all(difference - t):
                counter_within_t += 1
        if counter_within_t > best_counter:
            best_model = potential_model

    # remember the potential model and how many inliers it gave
    # do steps 1, 2, 3 for k times (k = parameter of number of iterations of alg)
    # choose the best model and return this

    return best_model
