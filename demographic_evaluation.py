import pandas as pd
from pickle import load
import pre_processing_functions
import numpy as np
from tensorflow.keras.models import load_model
import evaluation_functions


''' Demographic. '''
data_demographic = pd.read_csv('demographic_test.csv')

# Handling missing values.
column_types = data_demographic[data_demographic.columns.difference(['event_id'])].columns.to_series().groupby(data_demographic.dtypes).groups
column_types = {k.name: v for k, v in column_types.items()}
categorical_columns = column_types['object']
integer_columns = column_types['int64']
continous_columns = column_types['float64']

imputer_mf = load(open('imputer_mf.pkl', 'rb'))
data_demographic[categorical_columns] = imputer_mf.fit_transform(data_demographic[categorical_columns])

imputer_med = load(open('imputer_med.pkl', 'rb'))
data_demographic[integer_columns] = imputer_med.fit_transform(data_demographic[integer_columns])

imputer_mean = load(open('imputer_mean.pkl', 'rb'))
data_demographic[continous_columns] = imputer_mean.fit_transform(data_demographic[continous_columns])

# One-hot encoding.
encoder = load(open('onehotencoder_demographic.pkl', 'rb'))

categories = []
for i in categorical_columns:
    categories.append(data_demographic[i].unique())
longest_categories = np.array([max(categories[0], key=len), max(categories[1], key=len),max(categories[2], key=len), max(categories[3], key=len)]).reshape(-1, 4)
encoder.fit(longest_categories)

dummies = encoder.transform(data_demographic[categorical_columns]).toarray()
dummy_names = encoder.get_feature_names(categorical_columns)
data_demographic = pd.concat([data_demographic, pd.DataFrame(dummies, columns = dummy_names)], axis=1)
data_demographic = data_demographic.drop(categorical_columns, axis=1)

# Standardizing.
scaler = load(open('scaler.pkl', 'rb'))
data_demographic[[*integer_columns, *continous_columns]] = scaler.fit_transform(data_demographic[[*integer_columns, *continous_columns]])

test_x = data_demographic.sort_values(by=['event_id'])
test_x = test_x.drop(['event_id'], axis=1)
test_x = test_x.to_numpy()


''' Purchase events. '''
data_events = pd.read_csv('purchase_events_test.csv')

test_y = pre_processing_functions.binarize_and_split_purchases(data_events, split=False)[0]


''' Filter. '''
data_filter = pd.read_csv('filter_test.csv')

test_w = pre_processing_functions.binarize_and_split_purchases(data_filter, split=False)[0]


''' Predict. '''
model = load_model('model_demographic.h5')
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
statistical_significans.to_csv('statistical_significans_demographic.csv', index=False)

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
varying_thresholds.to_csv('varying_thresholds_demographic.csv', index=False)
