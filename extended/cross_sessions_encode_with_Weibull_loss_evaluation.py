import pandas as pd
import pre_processing_functions
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Masking
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
import numpy.ma as ma
import evaluation_functions


''' Time. '''
data_sessions = pd.read_csv('sessions_test.csv')
data_events = pd.read_csv('purchase_events_test.csv')

data_time = pre_processing_functions.time(data_sessions, data_events, train=False)


''' Sessions. '''
data_sessions_time = pre_processing_functions.sessions_time(data_sessions, data_time, train=False)
data_sessions_time['event_id'] = data_sessions_time[['max_event_time', 'event_number', 'user_id']].apply(tuple,axis=1).rank(method='dense').astype(int)
data_sessions_time = data_sessions_time.drop(['max_event_time', 'event_number', 'user_id'], axis=1)

data_sessions_time, dummy_names = pre_processing_functions.one_hot_encode_actions_fit_transform(data_sessions_time)

# Max pooling operation   
agg_dict = dict.fromkeys(dummy_names, 'max') 
data_sessions_time = data_sessions_time.groupby(['event_id', 'session_start'], as_index=False).agg(agg_dict)          

group_columns = ['event_id']
sort_columns = ['session_start']
n_steps = 22
test_x = pre_processing_functions.start_padding_and_split_sessions(data_sessions_time, group_columns, sort_columns, n_steps, split=False)


''' Purchase events. '''
data_events_time = pre_processing_functions.events_time(data_events, data_time, train=False)
data_events_time['event_id'] = data_events_time[['max_event_time', 'event_number', 'user_id']].apply(tuple,axis=1).rank(method='dense').astype(int)
data_events_time = data_events_time.drop(['max_event_time', 'event_number', 'user_id'], axis=1)

data_events_time, data_events_last = pre_processing_functions.compute_time_and_indicator(data_events_time)

group_columns = ['event_id']
sort_columns = ['session_start']
n_steps = 22
test_y = pre_processing_functions.start_padding_and_split_sessions(data_events_time, group_columns, sort_columns, n_steps, split=False)
last_y = pre_processing_functions.start_padding_and_split_sessions(data_events_last, group_columns, sort_columns, n_steps, split=False)

test_y = test_y.reshape((test_y.shape[0],test_y.shape[1],round((test_y.shape[2]-1)/2),2))


''' Filter. '''
data_filter = pd.read_csv('filter_test.csv')

data_filter_time = pre_processing_functions.filter_time(data_filter, data_time, train=False)
data_filter_time['event_id'] = data_filter_time[['max_event_time', 'event_number', 'user_id']].apply(tuple,axis=1).rank(method='dense').astype(int)
data_filter_time = data_filter_time.drop(['max_event_time', 'event_number', 'user_id'], axis=1)

data_filter_time = pre_processing_functions.binarize_filter(data_filter_time)

group_columns = ['event_id']
sort_columns = ['session_start']
n_steps = 22
test_w = pre_processing_functions.start_padding_and_split_sessions(data_filter_time, group_columns, sort_columns, n_steps, split=False)
test_w[test_w == 0] = np.nan


''' Predict. '''
# Model.
def output_lambda(x, init_alpha, max_beta_value=5.0, max_alpha_value=None):
    x = K.reshape(x, shape=(-1,K.int_shape(x)[1],round(K.int_shape(x)[2]/2),2))
    
    a = x[..., 0]
    b = x[..., 1]
    
    # Implicitly initialize alpha:
    if max_alpha_value is None:
        a = init_alpha * K.exp(a)
    else:
        a = init_alpha * K.clip(x=a, min_value=K.epsilon(),
                                max_value=max_alpha_value)
    
    m = max_beta_value
    if m > 1.05:  # some value >>1.0
        # shift to start around 1.0
        # assuming input is around 0.0
        _shift = np.log(m - 1.0)
    
        b = K.sigmoid(b - _shift)
    else:
        b = K.sigmoid(b)
    
    # Clipped sigmoid : has zero gradient at 0,1
    # Reduces the small tendency of instability after long training
    # by zeroing gradient.
    b = m * K.clip(x=b, min_value=K.epsilon(), max_value=1. - K.epsilon())
    
    x = K.stack([a, b], axis=-1)

    return x

init_alpha_concat = np.ones(16)

epochs, batch_size, units, rate = 100, 16, 64, 0.3
n_timesteps, n_features, n_outputs = test_x.shape[1], test_x.shape[2], test_y.shape[2]
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(n_timesteps,n_features)))
model.add(GRU(units, return_sequences=True))
model.add(Dropout(rate))
model.add(Dense(units, activation='relu'))
model.add(Dense(n_outputs*2))
model.add(Lambda(output_lambda, arguments={"init_alpha":init_alpha_concat, "max_beta_value":9.0}))


model.load_weights('model_encode_Weibull.h5')
pred = model.predict(test_x)

def weibull_quantiles(a, b, p):
    return a*np.power(-np.log(1.0-p),1.0/b)

pred_time = weibull_quantiles(pred[:,:,:,0], pred[:,:,:,1], 0.5)
test_w[test_w == 0] = np.nan
pred_time = pred_time * test_w
pred_time = np.where(np.isnan(pred_time), ma.array(pred_time, mask=np.isnan(pred_time)).max(axis=-1)[:, :, np.newaxis]+1, pred_time)


last_y = np.argwhere(last_y>0)
last_y = np.split(last_y[:,1], np.unique(last_y[:, 0], return_index=True)[1][1:])

pred_last = []
for i in range(len(last_y)):
    pred_last.append(pred_time[i, last_y[i], :])
pred_last = np.concatenate(pred_last)

test_y_last = []
for i in range(len(last_y)):
    test_y_last.append(test_y[i, last_y[i], :, 0])
test_y_last = np.concatenate(test_y_last)


''' Evaluation. '''
k = 3

hit =  evaluation_functions.hit(-pred_last, test_y_last, k)

precision = evaluation_functions.precision(-pred_last, test_y_last, k)

recall = evaluation_functions.recall(-pred_last, test_y_last, k)

rr = evaluation_functions.reciprocal_rank(-pred_last, test_y_last, k)

ap = evaluation_functions.average_precision(-pred_last, test_y_last, k)

print([np.mean(hit), np.mean(precision), np.mean(recall), np.mean(rr), np.mean(ap)])


# Statistical significans
statistical_significans = pd.DataFrame({'hit' : hit, 'precision' : precision, 'recall' : recall, 'RR' : rr, 'AP' : ap})
statistical_significans.to_csv('statistical_significans_encode_Weibull.csv', index=False)

# Varying thresholds
hr = []
precision = []
recall = []
mrr = []
mean_average_precision = []
for k in range(1,6):
    hr.append(np.mean(evaluation_functions.hit(-pred_last, test_y_last, k)))
    precision.append(np.mean(evaluation_functions.precision(-pred_last, test_y_last, k)))
    recall.append(np.mean(evaluation_functions.recall(-pred_last, test_y_last, k)))
    mrr.append(np.mean(evaluation_functions.reciprocal_rank(-pred_last, test_y_last, k)))
    mean_average_precision.append(np.mean(evaluation_functions.average_precision(-pred_last, test_y_last, k)))

varying_thresholds = pd.DataFrame({'HR' : hr, 'precision' : precision, 'recall' : recall, 'MRR' : mrr, 'MAP' : mean_average_precision})
varying_thresholds.to_csv('varying_thresholds_encode_Weibull.csv', index=False)
