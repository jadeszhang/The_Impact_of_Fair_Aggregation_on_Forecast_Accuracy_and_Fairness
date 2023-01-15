
import pandas as pd
import numpy 
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from distance_and_fairness import *
from sklearn_extra.cluster import KMedoids
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
# df_enrollment = pd.read_csv('C:/Users/jz839/OneDrive/research/Forecasting Paper/Data/admission_and_enrollment/enrollment/model_1.csv')
# df_admission = pd.read_csv('C:/Users/jz839/OneDrive/research/Forecasting Paper/Data/admission_and_enrollment/admission/model_1.csv')


def hirarchical_forecast (df, agg_level, disagg_level, aggregation_match):
    ''' method that performs hirarchical_forecast. 
    Parameters: df - dataframe that needs to be forecasted
                agg_level - list of aggregate columns
                disagg_level - list of disagg_level
                aggregation match - dictionary of matches of agg-disagg groups (e.g., )'''

    
    # add total column for df if it doesnt exist
    if 'total' not in df.columns:
        df['total'] = df[agg_level].sum(axis=1)

    # forecast for the aggregate groups - train set 20; test set 5
    pred_results = []
    for agg_cl in agg_level:
         train = df[agg_cl].iloc[0:-5]
         model =  ExponentialSmoothing(train, trend = True)
         fit = model.fit()
         pred = fit.forecast(5)
         df_pred = pd.DataFrame(pred)
         df_pred = df_pred.rename({'predicted_mean':agg_cl}, axis = 1)
         pred_results.append(df_pred)
    
    # merge the the aggregate groups together
    pred = pd.concat(pred_results, axis=1)
    
    # get the forecast of total
    pred['total'] = pred[agg_level].sum(axis=1)
    # top-down to forecast the disaggregate races
    for agg in agg_levels:
        for disagg in aggregation_match[agg]:
            prop = df[disagg].mean()/df[agg].mean()
            pred[disagg] = prop*pred[agg]

    
    return pred 

    
def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices


def distance1(fairness_matrix, sts_matrix, a , normalize = True):
    # calculate distance d = sts + alpha * fairness

    if normalize == True: 
        fairness_matrix = norm(fairness_matrix)
        sts_matrix = norm(sts_matrix)
    
    fairness_matrix = numpy.multiply(a, fairness_matrix)
    distance_m = numpy.subtract(sts_matrix,fairness_matrix)
    distance_m = norm(distance_m)
    return distance_m    



##############################################################
##############################################################
##############################################################



df_enrollment = pd.read_csv('C:/Users/Shuyu/Documents/research/OneDrive/research/Forecasting Paper/Data/admission_and_enrollment/enrollment/model_1.csv')
df_admission = pd.read_csv('C:/Users/Shuyu/Documents/research/OneDrive/research/Forecasting Paper/Data/admission_and_enrollment/admission/model_1.csv')

cols = list(df_enrollment.columns)
cols.remove('year')
cols.remove('Unnamed: 0')
df_enrollment = df_enrollment[cols]
df_admission = df_admission[cols]
corr_enrollment = numpy.array(df_enrollment.corr())

array_enrollment = df_enrollment.T.values
array_admission = df_admission.T.values
array_enrollment_std = (array_enrollment - array_enrollment.mean())/array_enrollment.std()
array_enrollment_nor = (array_enrollment - array_enrollment.min())/(array_enrollment.max() - array_enrollment.min())
fair_m = fairness_matrix(array_enrollment, array_admission, ts_reduction="mean")
sts_m = sts_matrix(array_enrollment)

# aggregation 
# for a in range(0,1,0.01):
# calculate distance d = sts - alpha * fairness
distance_m = distance1(fair_m, sts_m, a = 100, normalize=False)
kmedoids = KMedoids(n_clusters=3, random_state=0,max_iter = 5000, metric = 'precomputed').fit(distance_m)


###################################################################
###################################################################
################################## need to write the bisecting tree ######################



###########################################################################################
####################################################################
####################################################################




############# suppose we have an array of clusters ###############
aggregation_match = {}
agg_levels = []
labels = array(kmedoids.labels_)
clusters = list(unique(labels))
for c in clusters: 
    indexes = find_indices(labels, c)
    agg_race_name = 'a_race_' + str(c+1)
    agg_levels.append(agg_race_name)
    disagg_race_set = []
    for i in indexes: 
        disagg_race_name = 'race_' + str(i+1)
        disagg_race_set.append(disagg_race_name)
    aggregation_match[agg_race_name] = disagg_race_set

match = {w: k for k, v in aggregation_match.items() for w in v}
df_enrollment_model_3 = df_enrollment.groupby(by=match,axis=1).sum() 
df_enrollment_model_3 = pd.concat([df_enrollment_model_3, df_enrollment], axis=1, join="inner")
# make it time series
index = pd.date_range(start="1995-01-01", end="2020", freq="A-Jan")
df_enrollment_model_3 = df_enrollment_model_3.set_index(index)


# get the columns of levels
disagg_levels = [ele for ele in df_enrollment_model_3.columns if ele not in agg_levels]

# forecast 
df_enrollment_model_3_forecast = hirarchical_forecast(df_enrollment_model_3, agg_levels, disagg_levels, aggregation_match)
df_enrollment_model_3['total'] = df_enrollment_model_3[agg_levels].sum(axis=1)



#####################################################################
################################# Evaluation ########################
#####################################################################









# cluster = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average',distance_threshold=0)
# cluster = AgglomerativeClustering(n_clusters = 5, affinity='precomputed', linkage='average')
# cluster.fit(distance_m)
# for x in np.arange(0.01, 1, 0.001):
#     clustering = DBSCAN(eps = x, metric='precomputed')
#     clustering.fit(distance_m)
#     print(x)
#     print(clustering.labels_)
#     print()
#plot_dendrogram(cluster, truncate_mode="level")


# def plot_dendrogram(model, **kwargs):
#     # Create linkage matrix and then plot the dendrogram

#     # create the counts of samples under each node
#     counts = np.zeros(model.children_.shape[0])
#     n_samples = len(model.labels_)
#     for i, merge in enumerate(model.children_):
#         current_count = 0
#         for child_idx in merge:
#             if child_idx < n_samples:
#                 current_count += 1  # leaf node
#             else:
#                 current_count += counts[child_idx - n_samples]
#         counts[i] = current_count

#     linkage_matrix = np.column_stack(
#         [model.children_, model.distances_, counts]
#     ).astype(float)

#     # Plot the corresponding dendrogram
#     dendrogram(linkage_matrix, **kwargs)







