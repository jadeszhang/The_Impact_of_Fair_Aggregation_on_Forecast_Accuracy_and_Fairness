# This file is used to make the distance matrix



###################### fairness 
# P(enrollment|admission) = P(enrollment)/P(admission)

from numpy import *
    
def norm(matrix_a):
    mat_max, mat_min = matrix_a.max(), matrix_a.min()
    matrix_a = (matrix_a - mat_min)/(mat_max - mat_min)
    return matrix_a

def fairness_matrix(matrix_a, *attributes, ts_reduction="mean"):
    # calculate the conditional probs for every group 
    if (ts_reduction == "mean"):
        fair_array = matrix_a.mean(axis = 1)/attributes[0].mean(axis = 1)
    elif (ts_reduction == "last"):
        fair_array = []
        for i in range(0,len(matrix_a)):
            lst_ele = matrix_a[i][-1]/attributes[0][i][-1]
            fair_array.append(lst_ele)
    fair_array = array(fair_array)
    # make a fairness matrix (A_ij = |P_i - P_j|)
    fair_m = zeros((len(matrix_a),len(matrix_a)))
    for i in range(0, len(matrix_a)):
        for j in range(0, len(matrix_a)):
            fair_m[i][j] = abs(fair_array[i] - fair_array[j])
    return fair_m


def sts_matrix(matrix_a, time_period = 1):
    # calculate the sts matrix
    sts_m = zeros((len(matrix_a),len(matrix_a)))
    for i in range(0, len(matrix_a)):
        for j in range(0,len(matrix_a)):
            sts_sum = []
            for k in range(0,len(matrix_a[j])-1):
                # sts corresponds to the square root of the sum of the squared 
                # differences of the slopes obtained by considering time-series as linear 
                # functions between measurements
                sts_1 = (matrix_a[i][k+1] - matrix_a[i][k])/time_period
                sts_2 = (matrix_a[j][k+1] - matrix_a[j][k])/time_period
                sts = abs(sts_1 - sts_2)
                sts_sum.append(sts)
            sts_m[i][j] = sum(sts_sum)
    return sts_m


def distance(fairness_matrix, sts_matrix, a , normalize = True):
    # calculate distance d = sts + alpha * fairness

    if normalize == True: 
        fairness_matrix = norm(fairness_matrix)
        sts_matrix = norm(sts_matrix)
    
    fairness_matrix = multiply(a, fairness_matrix)
    distance_m = sts_matrix-fairness_matrix
    distance_m = norm(distance_m)
    return distance_m
