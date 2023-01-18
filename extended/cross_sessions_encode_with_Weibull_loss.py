import pandas as pd
import pre_processing_functions
from numpy.random import seed
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
import numpy.ma as ma
from sksurv.metrics import concordance_index_censored


''' Time. '''
data_sessions = pd.read_csv('sessions_train.csv')
data_events = pd.read_csv('purchase_events_train.csv')

data_time = pre_processing_functions.time(data_sessions, data_events)


''' Sessions. '''
data_sessions_time = pre_processing_functions.sessions_time(data_sessions, data_time)
data_sessions_time['event_id'] = data_sessions_time[['max_event_time', 'user_id']].apply(tuple,axis=1).rank(method='dense').astype(int)
data_sessions_time = data_sessions_time.drop(['max_event_time', 'user_id'], axis=1)

data_sessions_time, dummy_names = pre_processing_functions.one_hot_encode_actions_fit_transform(data_sessions_time)

# Max pooling operation   
agg_dict = dict.fromkeys(dummy_names, 'max') 
data_sessions_time = data_sessions_time.groupby(['event_id', 'valid', 'session_start'], as_index=False).agg(agg_dict)          

group_columns = ['event_id']
sort_columns = ['session_start']
n_steps = 22
train_x, valid_x = pre_processing_functions.start_padding_and_split_sessions(data_sessions_time, group_columns, sort_columns, n_steps)



''' Purchase events. '''
data_events_time = pre_processing_functions.events_time(data_events, data_time)
data_events_time['event_id'] = data_events_time[['max_event_time', 'user_id']].apply(tuple,axis=1).rank(method='dense').astype(int)
data_events_time = data_events_time.drop(['max_event_time', 'user_id'], axis=1)

data_events_time = pre_processing_functions.compute_time_and_indicator(data_events_time)[0]

group_columns = ['event_id']
sort_columns = ['session_start']
n_steps = 22
train_y, valid_y = pre_processing_functions.start_padding_and_split_sessions(data_events_time, group_columns, sort_columns, n_steps)

train_y = train_y.reshape((train_y.shape[0],train_y.shape[1],round((train_y.shape[2]-1)/2),2))
valid_y = valid_y.reshape((valid_y.shape[0],valid_y.shape[1],round((valid_y.shape[2]-1)/2),2))


''' Filter. '''
data_filter = pd.read_csv('filter_train.csv')

data_filter_time = pre_processing_functions.filter_time(data_filter, data_time)
data_filter_time['event_id'] = data_filter_time[['max_event_time', 'user_id']].apply(tuple,axis=1).rank(method='dense').astype(int)
data_filter_time = data_filter_time.drop(['max_event_time', 'user_id'], axis=1)

data_filter_time = pre_processing_functions.binarize_filter(data_filter_time)

group_columns = ['event_id']
sort_columns = ['session_start']
n_steps = 22
valid_w = pre_processing_functions.start_padding_and_split_sessions(data_filter_time, group_columns, sort_columns, n_steps)[1]
valid_w[valid_w == 0] = np.nan


''' Model. '''
seed(42)
tf.random.set_seed(42)


# Weibull loss and output activation function.
def weibull_loss_discrete(y_true, y_pred, name=None):  
    u = y_true[..., 0]
    y = y_true[..., 1]
    a = y_pred[..., 0]
    b = y_pred[..., 1]
    
    hazard0 = K.pow((y + 1e-35) / a, b)
    hazard1 = K.pow((y + 1.0) / a, b)
    
    mask = K.equal(y, 0.0) 
    mask = K.all(mask, axis=-1)
    mask = K.cast(mask, dtype=K.floatx())
    mask = 1 - mask
    mask = K.expand_dims(mask)
    
    loglikelihoods = (u * K.log(K.exp(hazard1 - hazard0) - 1.0) - hazard1) * mask

    loglikelihood = K.sum(loglikelihoods, axis=[1,2])
    
    loss = -1 * (K.mean(loglikelihood))
    return loss


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


# Fit.
mask_train = np.equal(train_y[:,:,:,1], 0.0) 
mask_train = np.all(mask_train, axis=-1)
mask_train = mask_train.astype(float)
mask_train = 1 - mask_train

init_alpha_concat = []
for i in range(train_y.shape[2]):
    tte_mean_train = np.average(train_y[:,:,i,1], weights=mask_train)
    init_alpha = -1.0/np.log(1.0-1.0/(tte_mean_train+1.0))
    init_alpha_concat.append(init_alpha)

del mask_train


def weibull_quantiles(a, b, p):
    return a*np.power(-np.log(1.0-p),1.0/b)

mask_valid = np.equal(valid_y[:,:,:,1], 0.0) 
mask_valid = np.all(mask_valid, axis=-1)
mask_valid = mask_valid.astype(float)
mask_valid = 1 - mask_valid


epochs, batch_size, units, rate = 100, 16, 64, 0.3
n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[2]
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(n_timesteps,n_features)))
model.add(GRU(units, return_sequences=True))
model.add(Dropout(rate))
model.add(Dense(units, activation='relu'))
model.add(Dense(n_outputs*2))
model.add(Lambda(output_lambda, arguments={"init_alpha":init_alpha_concat, "max_beta_value":9.0}))
model.compile(loss=weibull_loss_discrete, optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('model_encode_Weibull.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=True)

history = model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=epochs, batch_size=batch_size, callbacks=[es, mc])


model.load_weights('model_encode_Weibull.h5')
valid_pred = model.predict(valid_x)


valid_pred_time = weibull_quantiles(valid_pred[:,:,:,0], valid_pred[:,:,:,1], 0.5)
valid_pred_time = valid_pred_time * valid_w
valid_pred_time = np.where(np.isnan(valid_pred_time), ma.array(valid_pred_time, mask=np.isnan(valid_pred_time)).max(axis=-1)[:, :, np.newaxis]+1, valid_pred_time)


ci = np.empty((valid_y.shape[0], valid_y.shape[1]))
for i in range(valid_y.shape[0]):
    for j in range(valid_y.shape[1]):
        if mask_valid[i,j] == 0:
            ci[i,j] = np.nan
        else:
            ci[i,j] = 1-concordance_index_censored(valid_y[i,j,:,0].astype(bool), valid_y[i,j,:,1], valid_pred_time[i,j,:])[0]
np.nanmean(ci)