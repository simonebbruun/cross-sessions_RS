import pandas as pd
import pre_processing_functions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GRU
from keras.layers import Layer
import keras.backend as K
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
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        mask = K.equal(x, 0.0) 
        mask = K.all(mask, axis=-1)
        mask = K.cast(mask, dtype=K.floatx())
        mask = 1 - mask

        e = K.exp(e) * mask
        alpha = e/K.sum(e, axis=-1, keepdims=True)
        # alpha = K.softmax(e)
        
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context


epochs, batch_size, units, rate = 100, 32, 64, 0.3
n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
model_input = Input(shape=(n_timesteps,n_features))

model = Masking(mask_value=0.0)(model_input)
model = GRU(units, return_sequences=True)(model)
model = attention()(model)
model = Dropout(rate)(model)
model = Dense(units, activation='relu')(model)
model = Dense(n_outputs, activation='sigmoid')(model)

model = Model(inputs=model_input, outputs=model)

model.compile(loss='binary_crossentropy', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('model_encode_attention.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history = model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=epochs, batch_size=batch_size, callbacks=[es, mc])

saved_model = load_model('model_encode_attention.h5', custom_objects = {"attention": attention})
valid_pred = saved_model.predict(valid_x)
valid_pred = valid_pred*valid_w
auc = metrics.roc_auc_score(valid_y, valid_pred)
