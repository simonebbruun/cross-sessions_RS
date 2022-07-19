import pandas as pd
import pre_processing_functions
import numpy as np
import evaluation_functions


''' Purchase events. '''
data_events = pd.read_csv('purchase_events_test.csv')

test_y = pre_processing_functions.binarize_and_split_purchases(data_events, split=False)[0]


''' Filter. '''
data_filter = pd.read_csv('filter_test.csv')

test_w = pre_processing_functions.binarize_and_split_purchases(data_filter, split=False)[0]


''' Predict. '''
n_obs = test_y.shape[0]
n_items = test_y.shape[1]

pred = np.empty([n_obs*n_items]).reshape(n_obs, n_items)
for i in range(n_items):
    pred[:,i] = np.random.uniform(0,1,n_obs)
pred = pred*test_w


''' Evaluation. '''
k = 3

hit =  evaluation_functions.hit(pred, test_y, k)

precision = evaluation_functions.precision(pred, test_y, k)

recall = evaluation_functions.recall(pred, test_y, k)

rr = evaluation_functions.reciprocal_rank(pred, test_y, k)

ap = evaluation_functions.average_precision(pred, test_y, k)

print([np.mean(hit), np.mean(precision), np.mean(recall), np.mean(rr), np.mean(ap)])


# Statistical significans
statistical_significans = pd.DataFrame({'hit' : hit, 'precision' : precision, 'recall' : recall, 'RR' : rr, 'AP' : ap})
statistical_significans.to_csv('statistical_significans_random.csv', index=False)

# Varying thresholds
hr = []
precision = []
recall = []
mrr = []
mean_average_precision = []
for k in range(1,6):
    hr.append(np.mean(evaluation_functions.hit(pred, test_y, k)))
    precision.append(np.mean(evaluation_functions.precision(pred, test_y, k)))
    recall.append(np.mean(evaluation_functions.recall(pred, test_y, k)))
    mrr.append(np.mean(evaluation_functions.reciprocal_rank(pred, test_y, k)))
    mean_average_precision.append(np.mean(evaluation_functions.average_precision(pred, test_y, k)))

varying_thresholds = pd.DataFrame({'HR' : hr, 'precision' : precision, 'recall' : recall, 'MRR' : mrr, 'MAP' : mean_average_precision})
varying_thresholds.to_csv('varying_thresholds_random.csv', index=False)
