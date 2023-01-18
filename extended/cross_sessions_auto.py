import numpy as np
import pandas as pd
import pre_processing_functions
from numpy.random import seed
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import evaluation_functions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from sklearn import metrics


''' Sessions. '''
data_sessions = pd.read_csv('sessions_train.csv')
n_sections = len(pd.unique(data_sessions['action_section']))
n_objects = len(pd.unique(data_sessions['action_object']))
n_types = len(pd.unique(data_sessions['action_type']))

data_sessions = pre_processing_functions.one_hot_encode_actions_fit_transform(data_sessions)[0]


''' Sessions for autoencoder. '''
data_sessions_auto = data_sessions.drop(['event_id'], axis=1)
data_sessions_auto = data_sessions_auto.drop_duplicates()

group_columns = ['session_id']
sort_columns = ['action_time']
n_steps = 30
train_x, valid_x = pre_processing_functions.start_padding_and_split_sessions(data_sessions_auto, group_columns, sort_columns, n_steps)
train_y, valid_y = pre_processing_functions.end_padding_and_split_sessions(data_sessions_auto, group_columns, sort_columns, n_steps)

train_section = train_y[:,:,0:n_sections].astype(np.int8)
train_object = train_y[:,:,n_sections:(n_sections+n_objects)].astype(np.int8)
train_type = train_y[:,:,(n_sections+n_objects):(n_sections+n_objects+n_types)].astype(np.int8)

valid_section = valid_y[:,:,0:n_sections].astype(np.int8)
valid_object = valid_y[:,:,n_sections:(n_sections+n_objects)].astype(np.int8)
valid_type = valid_y[:,:,(n_sections+n_objects):(n_sections+n_objects+n_types)].astype(np.int8)


train_sample_weight = np.zeros(train_y.shape[0]*n_steps, dtype=np.int8).reshape(train_y.shape[0], n_steps)
for i in range(train_y.shape[0]):
    for j in range(n_steps):
        train_sample_weight[i, j] = np.where(np.sum(train_y[i, j, :]) == 0, 0, 1)

valid_sample_weight = np.zeros(valid_y.shape[0]*n_steps, dtype=np.int8).reshape(valid_y.shape[0], n_steps)
for i in range(valid_y.shape[0]):
    for j in range(n_steps):
        valid_sample_weight[i, j] = np.where(np.sum(valid_y[i, j, :]) == 0, 0, 1)


''' Autoencoder model. '''
seed(42)
tf.random.set_seed(42)

epochs, batch_size, units = 300, 128, 512
n_timesteps, n_features, n_output1, n_output2, n_output3 = train_x.shape[1], train_x.shape[2], train_section.shape[2], train_object.shape[2], train_type.shape[2]

tf_train = tf.data.Dataset.from_tensor_slices((train_x, (train_section, train_object, train_type), (train_sample_weight, train_sample_weight, train_sample_weight)))
tf_train = tf_train.cache()
tf_train = tf_train.shuffle(train_x.shape[0])
tf_train = tf_train.batch(batch_size, drop_remainder=True)
tf_train = tf_train.prefetch(tf.data.AUTOTUNE)

tf_valid = tf.data.Dataset.from_tensor_slices((valid_x, (valid_section, valid_object, valid_type), (valid_sample_weight, valid_sample_weight, valid_sample_weight)))
tf_valid = tf_valid.cache()
tf_valid = tf_valid.batch(valid_x.shape[0], drop_remainder=True)
tf_valid = tf_valid.prefetch(tf.data.AUTOTUNE)

input_x = Input(shape=(n_timesteps,n_features))

masked = Masking(mask_value=0.0)(input_x) 
encoder = GRU(units, return_sequences=False)(masked)
decoder1 = RepeatVector(n_timesteps)(encoder)
decoder2 = GRU(units, return_sequences=True)(decoder1)

decoder_section = TimeDistributed(Dense(n_output1, activation='softmax'))(decoder2)
decoder_object = TimeDistributed(Dense(n_output2, activation='softmax'))(decoder2)
decoder_type = TimeDistributed(Dense(n_output3, activation='softmax'))(decoder2)

model = Model(inputs=input_x, outputs=[decoder_section, decoder_object, decoder_type])

model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], optimizer='adam', sample_weight_mode='temporal')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('model_auto_autoencoder.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

model.fit(tf_train, validation_data=tf_valid, epochs=epochs, batch_size=batch_size, callbacks=[es, mc])

saved_model = load_model('model_auto_autoencoder.h5')
pred = saved_model.predict(valid_x, verbose=0)
proba_section = pred[0]
proba_object = pred[1]
proba_type = pred[2]

avg_weight = np.sum(valid_sample_weight, axis=0)

accuracy_section = evaluation_functions.autoencoder_accuracy(proba_section, n_timesteps, n_sections, valid_section, valid_sample_weight, avg_weight)
accuracy_object = evaluation_functions.autoencoder_accuracy(proba_object, n_timesteps, n_objects, valid_object, valid_sample_weight, avg_weight)
accuracy_type = evaluation_functions.autoencoder_accuracy(proba_type, n_timesteps, n_types, valid_type, valid_sample_weight, avg_weight)

encoder_model = Model(saved_model.input, saved_model.layers[-6].output)  
encoder_model.save('model_auto_encoder.h5')


''' Sessions for RNN. '''
group_columns = ['event_id', 'session_id']
sort_columns = ['action_time']
n_steps = 30
data_sessions_RNN = data_sessions.drop(['valid'], axis=1)
data_sessions_RNN = pre_processing_functions.start_padding_and_split_sessions(data_sessions_RNN, group_columns, sort_columns, n_steps, split=False)

encoder_pred = encoder_model.predict(data_sessions_RNN)

data_sessions = data_sessions[['event_id', 'valid', 'session_id', 'action_time']].groupby(['event_id', 'valid', 'session_id'], as_index=False).min()
data_sessions = data_sessions.sort_values(by=['event_id', 'session_id']).reset_index(drop=True)
data_sessions = pd.concat([data_sessions, pd.DataFrame(encoder_pred)], axis=1)
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


''' RNN model. '''
seed(42)
tf.random.set_seed(42)

epochs, batch_size, units, rate = 100, 32, 64, 0.4
n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(n_timesteps,n_features)))
model.add(GRU(units, return_sequences=False))
model.add(Dropout(rate))
model.add(Dense(units, activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('model_auto.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history = model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=epochs, batch_size=batch_size, callbacks=[es, mc])

saved_model = load_model('model_auto.h5')
valid_pred = saved_model.predict(valid_x)
valid_pred = valid_pred*valid_w
auc = metrics.roc_auc_score(valid_y, valid_pred)