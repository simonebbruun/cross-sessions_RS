import numpy as np
import pandas as pd
import pre_processing_functions
from numpy.random import seed
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn import metrics


''' Sessions. '''
data_sessions = pd.read_csv('sessions_train.csv')

data_sessions = pre_processing_functions.one_hot_encode_actions_fit_transform(data_sessions)[0]

# Add column with order of sessions for each purchase event.
data_sessions['session_start'] = data_sessions.groupby(['event_id', 'session_id']).action_time.transform(np.min)
data_sessions['session_order'] = data_sessions.sort_values(['session_start']) \
              .groupby(['event_id']) \
              .cumcount() + 1
data_sessions = data_sessions.drop(['session_id', 'session_start'], axis=1)    
# data_sessions = data_sessions.drop(['session_id', 'action_time'], axis=1)            

group_columns = ['event_id']
sort_columns = ['session_order', 'action_time']
n_steps = 222
train_x, valid_x = pre_processing_functions.start_padding_and_split_sessions(data_sessions, group_columns, sort_columns, n_steps)


''' Purchase events. '''
data_events = pd.read_csv('purchase_events_train.csv')

train_y, valid_y = pre_processing_functions.binarize_and_split_purchases(data_events)


''' Filter. '''
data_filter = pd.read_csv('filter_train.csv')

train_w, valid_w = pre_processing_functions.binarize_and_split_purchases(data_filter)


''' Model. '''
seed(42)
tf.random.set_seed(42)

epochs, batch_size, units, rate = 100, 128, 64, 0.3
n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(n_timesteps,n_features)))
model.add(GRU(units, return_sequences=False))
model.add(Dropout(rate))
model.add(Dense(units, activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('model_concat.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history = model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=epochs, batch_size=batch_size, callbacks=[es, mc])

saved_model = load_model('model_concat.h5')
valid_pred = saved_model.predict(valid_x)
valid_pred = valid_pred*valid_w
auc = metrics.roc_auc_score(valid_y, valid_pred)