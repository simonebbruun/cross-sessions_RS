import pandas as pd
import pre_processing_functions
import numpy as np
from keras.layers import Layer
import keras.backend as K
from tensorflow.keras.models import load_model
import evaluation_functions


''' Sessions. '''
data_sessions = pd.read_csv('sessions_test.csv')

data_sessions, dummy_names = pre_processing_functions.one_hot_encode_actions_transform(data_sessions)

# Max pooling operation   
agg_dict = dict.fromkeys(dummy_names, 'max')
agg_dict.update(dict.fromkeys(['action_time'], 'min'))   
data_sessions = data_sessions.groupby(['event_id', 'session_id'], as_index=False).agg(agg_dict)          
data_sessions = data_sessions.drop(['session_id'], axis=1)  

group_columns = ['event_id']
sort_columns = ['action_time']
n_steps = 7
test_x = pre_processing_functions.start_padding_and_split_sessions(data_sessions, group_columns, sort_columns, n_steps, split=False)


''' Purchase events. '''
data_events = pd.read_csv('purchase_events_test.csv')
data_events = data_events.drop(['user_id', 'event_time'], axis=1)

test_y = pre_processing_functions.binarize_and_split_purchases(data_events, split=False)[0]


''' Filter. '''
data_filter = pd.read_csv('filter_test.csv')

test_w = pre_processing_functions.binarize_and_split_purchases(data_filter, split=False)[0]


''' Predict. '''
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
    
model = load_model('model_encode_attention.h5', custom_objects = {"attention": attention})
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
statistical_significans.to_csv('statistical_significans_encode_attention.csv', index=False)

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
varying_thresholds.to_csv('varying_thresholds_encode_attention.csv', index=False)
