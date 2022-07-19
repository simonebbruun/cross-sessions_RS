import pandas as pd
import pre_processing_functions
import numpy as np
from tensorflow.keras.models import load_model
import evaluation_functions


''' Sessions. '''
data_sessions = pd.read_csv('sessions_test.csv')

data_sessions = pre_processing_functions.one_hot_encode_actions_transform(data_sessions)[0]

# Add column with start time for each session.
data_sessions['session_start'] = data_sessions.groupby(['event_id', 'session_id']).action_time.transform(np.min)

# Select the last session for each purchase events.
idx = data_sessions.groupby(['event_id'])['session_start'].transform(max) == data_sessions['session_start']
data_sessions = data_sessions[idx]

data_sessions = data_sessions.drop(['session_id', 'session_start'], axis=1)

session_length = data_sessions.groupby(['event_id']).count()
session_length = session_length.sort_values(by=['event_id'])
session_length = session_length.values[:,0]-1

group_columns = ['event_id']
sort_columns = ['action_time']
n_steps = 30
test_x = pre_processing_functions.end_padding_and_split_sessions(data_sessions, group_columns, sort_columns, n_steps, split=False)

test_x = test_x[:,:-1,:]

object_columns = [col for col in data_sessions if col.startswith('action_object')]


''' Purchase events. '''
data_events = pd.read_csv('purchase_events_test.csv')

test_y, item_columns = pre_processing_functions.binarize_and_split_purchases(data_events, split=False)
item_columns = 'action_object_' + item_columns

''' Filter. '''
data_filter = pd.read_csv('filter_test.csv')

test_w = pre_processing_functions.binarize_and_split_purchases(data_filter, split=False)[0]


''' Predict. '''
model = load_model('model_GRU4REC.h5')
preds = model.predict(test_x)
pred_last = np.empty([preds.shape[0], preds.shape[2]])
for i in range(len(preds)):
    pred_last[i] = preds[i:(i+1),(session_length[i]-1),:]
    
pred = pd.DataFrame(pred_last, columns=object_columns)

for i in item_columns:
    if i not in pred:
        pred[i] = 0
pred = pred[item_columns]

pred = pred.values*test_w


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
statistical_significans.to_csv('statistical_significans_GRU4REC.csv', index=False)

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
varying_thresholds.to_csv('varying_thresholds_GRU4REC.csv', index=False)
