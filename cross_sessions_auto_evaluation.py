import pandas as pd
import pre_processing_functions
import numpy as np
from tensorflow.keras.models import load_model
import evaluation_functions


''' Sessions. '''
data_sessions = pd.read_csv('sessions_test.csv')

data_sessions = pre_processing_functions.one_hot_encode_actions_transform(data_sessions)[0]

group_columns = ['event_id', 'session_id']
sort_columns = ['action_time']
n_steps = 30
data_sessions_encoder = pre_processing_functions.start_padding_and_split_sessions(data_sessions, group_columns, sort_columns, n_steps, split=False)

encoder_model = load_model('model_auto_encoder.h5')
encoder_pred = encoder_model.predict(data_sessions_encoder)

data_sessions = data_sessions[['event_id', 'session_id', 'action_time']].groupby(['event_id', 'session_id'], as_index=False).min()
data_sessions = data_sessions.sort_values(by=['event_id', 'session_id']).reset_index(drop=True)
data_sessions = pd.concat([data_sessions, pd.DataFrame(encoder_pred)], axis=1)
data_sessions = data_sessions.drop(['session_id'], axis=1) 

group_columns = ['event_id']
sort_columns = ['action_time']
n_steps = 7
test_x = pre_processing_functions.start_padding_and_split_sessions(data_sessions, group_columns, sort_columns, n_steps, split=False)


''' Purchase events. '''
data_events = pd.read_csv('purchase_events_test.csv')

test_y = pre_processing_functions.binarize_and_split_purchases(data_events, split=False)[0]


''' Filter. '''
data_filter = pd.read_csv('filter_test.csv')

test_w = pre_processing_functions.binarize_and_split_purchases(data_filter, split=False)[0]


''' Predict. '''
model = load_model('model_auto.h5')
pred = model.predict(test_x)
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
statistical_significans.to_csv('statistical_significans_auto.csv', index=False)

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
varying_thresholds.to_csv('varying_thresholds_auto.csv', index=False)
