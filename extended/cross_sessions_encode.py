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

data_sessions, dummy_names = pre_processing_functions.one_hot_encode_actions_fit_transform(data_sessions)

# Max pooling operation   
agg_dict = dict.fromkeys(dummy_names, 'max')
agg_dict.update(dict.fromkeys(['action_time'], 'min'))   
data_sessions = data_sessions.groupby(['event_id', 'valid', 'session_id'], as_index=False).agg(agg_dict)          
data_sessions = data_sessions.drop(['session_id'], axis=1)  

group_columns = ['event_id']
sort_columns = ['action_time']
n_steps = 7
train_x, valid_x = pre_processing_functions.start_padding_and_split_sessions(data_sessions, group_columns, sort_columns, n_steps)


''' Purchase events. '''
data_events = pd.read_csv('purchase_events_train.csv')
data_events = data_events.drop(['user_id', 'event_time'], axis=1)

train_y, valid_y = pre_processing_functions.binarize_and_split_purchases(data_events)


''' Filter. '''
data_filter = pd.read_csv('filter_train.csv')

train_w, valid_w = pre_processing_functions.binarize_and_split_purchases(data_filter)


''' Model. '''
seed(42)
tf.random.set_seed(42)

epochs, batch_size, units, rate = 100, 32, 64, 0.3
n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(n_timesteps,n_features)))
model.add(GRU(units, return_sequences=False))
model.add(Dropout(rate))
model.add(Dense(units, activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('model_encode.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history = model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=epochs, batch_size=batch_size, callbacks=[es, mc])

saved_model = load_model('model_encode.h5')
valid_pred = saved_model.predict(valid_x)
valid_pred = valid_pred*valid_w
auc = metrics.roc_auc_score(valid_y, valid_pred)
