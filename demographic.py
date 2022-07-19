import pandas as pd
from sklearn.impute import SimpleImputer
from pickle import dump
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pre_processing_functions
from numpy.random import seed
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn import metrics


''' Demographic. '''
data_demographic = pd.read_csv('demographic_train.csv')

# Handling missing values.
column_types = data_demographic[data_demographic.columns.difference(['event_id', 'valid'])].columns.to_series().groupby(data_demographic.dtypes).groups
column_types = {k.name: v for k, v in column_types.items()}
categorical_columns = column_types['object']
integer_columns = column_types['int64']
continous_columns = column_types['float64']

imputer_mf = SimpleImputer(missing_values=None, strategy='most_frequent')
data_demographic[categorical_columns] = imputer_mf.fit_transform(data_demographic[categorical_columns])

imputer_med = SimpleImputer(strategy='median')
data_demographic[integer_columns] = imputer_med.fit_transform(data_demographic[integer_columns])

imputer_mean = SimpleImputer()
data_demographic[continous_columns] = imputer_mean.fit_transform(data_demographic[continous_columns])

dump(imputer_mf, open('imputer_mf.pkl', 'wb'))
dump(imputer_med, open('imputer_med.pkl', 'wb'))
dump(imputer_mean, open('imputer_mean.pkl', 'wb'))

# One-hot encoding.
encoder = OneHotEncoder()
dummies = encoder.fit_transform(data_demographic[categorical_columns]).toarray()
dummy_names = encoder.get_feature_names(categorical_columns)
data_demographic = pd.concat([data_demographic, pd.DataFrame(dummies, columns = dummy_names)], axis=1)
data_demographic = data_demographic.drop(categorical_columns, axis=1)

encoder_categories = encoder.categories_
encoder_save = OneHotEncoder(handle_unknown="ignore", categories=encoder_categories)
dump(encoder_save, open('onehotencoder_demographic.pkl', 'wb'))

# Standardizing.
scaler = StandardScaler()
data_demographic[[*integer_columns, *continous_columns]] = scaler.fit_transform(data_demographic[[*integer_columns, *continous_columns]])
dump(scaler, open('scaler.pkl', 'wb'))

# Split.
train_x = data_demographic[data_demographic['valid']==0]
valid_x = data_demographic[data_demographic['valid']==1]

train_x = train_x.sort_values(by=['event_id'])
train_x = train_x.drop(['event_id', 'valid'], axis=1)
train_x = train_x.to_numpy()

valid_x = valid_x.sort_values(by=['event_id'])
valid_x = valid_x.drop(['event_id', 'valid'], axis=1)
valid_x = valid_x.to_numpy()


''' Purchase events. '''
data_events = pd.read_csv('purchase_events_train.csv')

train_y, valid_y = pre_processing_functions.binarize_and_split_purchases(data_events)


''' Filter. '''
data_filter = pd.read_csv('filter_train.csv')

train_w, valid_w = pre_processing_functions.binarize_and_split_purchases(data_filter)


''' Model. '''
seed(42)
tf.random.set_seed(42)

epochs, batch_size, units, rate = 100, 32, 32, 0.3
n_features, n_outputs = train_x.shape[1], train_y.shape[1]
model = Sequential()
model.add(Dense(units, input_dim=n_features))
model.add(Dropout(rate))
model.add(Dense(units, activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('model_demographic.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history = model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=epochs, batch_size=batch_size, callbacks=[es, mc])

saved_model = load_model('model_demographic.h5')
valid_pred = saved_model.predict(valid_x)
valid_pred = valid_pred*valid_w
auc = metrics.roc_auc_score(valid_y, valid_pred)