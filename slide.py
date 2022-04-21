import numpy
import math
import torch
from sklearn.cluster import KMeans

def slide_MTS_dim(X, alpha):
    num_of_old_MTS = numpy.shape(X)[0]
    dim_of_old_MTS = numpy.shape(X)[1]
    length_of_old_MTS = numpy.shape(X)[2]
    length_of_new_MTS = int(length_of_old_MTS * alpha)

    X_alpha = X[:, :, 0 : length_of_new_MTS]

    # determine step
    if (length_of_old_MTS <= 50) :
        step = 1
    elif (length_of_old_MTS > 50 and length_of_old_MTS <= 100):
        step = 2
    elif (length_of_old_MTS > 100 and length_of_old_MTS <= 300):
        step = 3
    elif (length_of_old_MTS > 300 and length_of_old_MTS <= 1000):
        step = 4
    elif (length_of_old_MTS > 1000 and length_of_old_MTS <= 1500):
        step = 5
    elif (length_of_old_MTS > 1500 and length_of_old_MTS <= 2000):
        step = 7
    elif (length_of_old_MTS > 2000 and length_of_old_MTS <= 3000):
        step = 10
    else:
        step = 20

    # determine step number
    step_num = int(math.ceil((length_of_old_MTS -length_of_new_MTS)/step))

    # still slide to 3D
    for k in range(1, length_of_old_MTS -length_of_new_MTS+1, step):

        X_temp = X[:, :, k : length_of_new_MTS + k]
        X_alpha = numpy.concatenate((X_alpha, X_temp), axis = 0)

    # numpy reshape 3D to 2D
    X_alpha = X_alpha.reshape(num_of_old_MTS * dim_of_old_MTS * (step_num+1), length_of_new_MTS)

    return X_alpha

def slide_MTS_dim_step(X, class_label, alpha):
    num_of_old_MTS = numpy.shape(X)[0]
    dim_of_old_MTS = numpy.shape(X)[1]
    length_of_old_MTS = numpy.shape(X)[2]
    length_of_new_MTS = int(length_of_old_MTS * alpha)

    #init variate label
    candidate_dim_init = [0]
    for m in range (1, dim_of_old_MTS):
        candidate_dim_init.append(m)

    X_alpha = X[:, :, 0 : length_of_new_MTS]

    # determine step
    if (length_of_old_MTS <= 50) :
        step = 1
    elif (length_of_old_MTS > 50 and length_of_old_MTS <= 100):
        step = 2
    elif (length_of_old_MTS > 100 and length_of_old_MTS <= 300):
        step = 3
    elif (length_of_old_MTS > 300 and length_of_old_MTS <= 1000):
        step = 4
    elif (length_of_old_MTS > 1000 and length_of_old_MTS <= 1500):
        step = 5
    elif (length_of_old_MTS > 1500 and length_of_old_MTS <= 2000):
        step = 7
    elif (length_of_old_MTS > 2000 and length_of_old_MTS <= 3000):
        step = 10
    else:
        step = 20

    # determine step number
    step_num = int(math.ceil((length_of_old_MTS -length_of_new_MTS)/step))

    # still slide to 3D
    for k in range(1, length_of_old_MTS -length_of_new_MTS+1, step):

        X_temp = X[:, :, k : length_of_new_MTS + k]
        X_alpha = numpy.concatenate((X_alpha, X_temp), axis = 0)

    # numpy reshape 3D to 2D
    X_alpha = X_alpha.reshape(num_of_old_MTS * dim_of_old_MTS * (step_num+1), length_of_new_MTS)

    # compose the final class label
    candidate_class_label = class_label.repeat((dim_of_old_MTS * (step_num+1)))

    # compose the final variate label
    candidate_dim = candidate_dim_init * (num_of_old_MTS * (step_num+1))

    return X_alpha, candidate_dim, candidate_class_label

def slide_MTS_tensor_step(X, alpha):
    num_of_old_MTS = X.size(0)
    dim_of_old_MTS = X.size(1)
    length_of_old_MTS = X.size(2)
    length_of_new_MTS = int(length_of_old_MTS * alpha)

    X_alpha = X[:, :, 0 : length_of_new_MTS]

    # determine step
    if (length_of_old_MTS <= 50) :
        step = 1
    elif (length_of_old_MTS > 50 and length_of_old_MTS <= 100):
        step = 2
    elif (length_of_old_MTS > 100 and length_of_old_MTS <= 300):
        step = 3
    elif (length_of_old_MTS > 300 and length_of_old_MTS <= 1000):
        step = 4
    elif (length_of_old_MTS > 1000 and length_of_old_MTS <= 1500):
        step = 5
    elif (length_of_old_MTS > 1500 and length_of_old_MTS <= 2000):
        step = 7
    elif (length_of_old_MTS > 2000 and length_of_old_MTS <= 3000):
        step = 10
    else:
        step = 20

    # determine step number
    step_num = int(math.ceil((length_of_old_MTS -length_of_new_MTS)/step))

    # still slide to 3D
    for k in range(1, length_of_old_MTS-length_of_new_MTS+1, step):

        X_temp = X[:, :, k : length_of_new_MTS + k]
        X_alpha = torch.cat((X_alpha, X_temp), dim = 0)

    # numpy reshape 3D to 2D
    X_beta = torch.reshape(X_alpha, (num_of_old_MTS * dim_of_old_MTS * (step_num+1), length_of_new_MTS))

    return X_beta
