import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from pickle import dump
from pickle import load
from sklearn.preprocessing import MultiLabelBinarizer

def one_hot_encode_actions_fit_transform(data):
    # Fill nan
    data['action_object'] = data['action_object'].fillna('no object')

    # One-hot encoding
    encoder = OneHotEncoder()
    dummies = encoder.fit_transform(data[['action_section','action_object','action_type']]).toarray()
    dummy_names = encoder.get_feature_names(['action_section','action_object','action_type'])
    data1 = pd.concat([data, pd.DataFrame(dummies, columns = dummy_names)], axis=1)
    data1 = data1.drop(['action_section','action_object','action_type'], axis=1)
    
    encoder_categories = encoder.categories_
    encoder_save = OneHotEncoder(handle_unknown="ignore", categories=encoder_categories)
    dump(encoder_save, open('onehotencoder_actions.pkl', 'wb'))
    
    return data1, dummy_names

def one_hot_encode_actions_transform(data):
    # Fill nan
    data['action_object'] = data['action_object'].fillna('no object')

    # One-hot encoding
    encoder = load(open('onehotencoder_actions.pkl', 'rb'))

    section_categories = data['action_section'].unique()
    object_categories = data['action_object'].unique()
    type_categories = data['action_type'].unique()
    longest_categories = np.array([max(section_categories, key=len), max(object_categories, key=len), max(type_categories, key=len)]).reshape(-1, 3)
    encoder.fit(longest_categories)

    dummies = encoder.transform(data[['action_section','action_object','action_type']]).toarray()
    dummy_names = encoder.get_feature_names(['action_section','action_object','action_type'])
    data1 = pd.concat([data, pd.DataFrame(dummies, columns = dummy_names)], axis=1)
    data1 = data1.drop(['action_section','action_object','action_type'], axis=1)

    return data1, dummy_names


def start_padding_and_split_sessions(data, group_columns, sort_columns, n_steps, split=True):
    if split:
        # Split
        train = data[data['valid']==0]
        valid = data[data['valid']==1]

        # Padding
        train = train.sort_values(by=[*group_columns, *sort_columns])
        train = train.drop(['valid', *sort_columns], axis=1)

        train_array = np.array(list(train.groupby(group_columns).apply(pd.DataFrame.to_numpy)))

        n_obs = len(train_array)
        n_columns = len(train.columns)

        train_padded = np.empty((n_obs,n_steps,n_columns), dtype = np.float32)
        for i in range(0,len(train_array)):
            train_padded[i] = np.pad(train_array[i], ((n_steps-len(train_array[i]),0), (0, 0)), 'constant')
            
        train_padded = train_padded[:, :, len(group_columns):n_columns]


        valid = valid.sort_values(by=[*group_columns, *sort_columns])
        valid = valid.drop(['valid', *sort_columns], axis=1)

        valid_array = np.array(list(valid.groupby(group_columns).apply(pd.DataFrame.to_numpy)))

        n_obs = len(valid_array)
        n_columns = len(valid.columns)

        valid_padded = np.empty((n_obs,n_steps,n_columns), dtype = np.float32)
        for i in range(0,len(valid_array)):
            valid_padded[i] = np.pad(valid_array[i], ((n_steps-len(valid_array[i]),0), (0, 0)), 'constant')
            
        valid_padded = valid_padded[:, :, len(group_columns):n_columns]
        return train_padded, valid_padded
    else:
        # Padding
        test = data.sort_values(by=[*group_columns, *sort_columns])
        test = test.drop([*sort_columns], axis=1)

        test_array = np.array(list(test.groupby(group_columns).apply(pd.DataFrame.to_numpy)))

        n_obs = len(test_array)
        n_columns = len(test.columns)

        test_padded = np.empty((n_obs,n_steps,n_columns), dtype = np.float32)
        for i in range(0,len(test_array)):
            test_padded[i] = np.pad(test_array[i], ((n_steps-len(test_array[i]),0), (0, 0)), 'constant')
            
        test_padded = test_padded[:, :, len(group_columns):n_columns]
        return test_padded
    
    
def end_padding_and_split_sessions(data, group_columns, sort_columns, n_steps, split=True):
    if split:
        # Split
        train = data[data['valid']==0]
        valid = data[data['valid']==1]

        # Padding
        train = train.sort_values(by=[*group_columns, *sort_columns])
        train = train.drop(['valid', *sort_columns], axis=1)

        train_array = np.array(list(train.groupby(group_columns).apply(pd.DataFrame.to_numpy)))

        n_obs = len(train_array)
        n_columns = len(train.columns)

        train_padded = np.empty((n_obs,n_steps,n_columns), dtype = np.float32)
        for i in range(0,len(train_array)):
            train_padded[i] = np.pad(train_array[i], ((0,n_steps-len(train_array[i])), (0, 0)), 'constant') 
            
        train_padded = train_padded[:, :, len(group_columns):n_columns]


        valid = valid.sort_values(by=[*group_columns, *sort_columns])
        valid = valid.drop(['valid', *sort_columns], axis=1)

        valid_array = np.array(list(valid.groupby(group_columns).apply(pd.DataFrame.to_numpy)))

        n_obs = len(valid_array)
        n_columns = len(valid.columns)

        valid_padded = np.empty((n_obs,n_steps,n_columns), dtype = np.float32)
        for i in range(0,len(valid_array)):
            valid_padded[i] = np.pad(valid_array[i], ((0,n_steps-len(valid_array[i])), (0, 0)), 'constant') 
            
        valid_padded = valid_padded[:, :, len(group_columns):n_columns]
        return train_padded, valid_padded
    else:
        # Padding
        test = data.sort_values(by=[*group_columns, *sort_columns])
        test = test.drop([*sort_columns], axis=1)

        test_array = np.array(list(test.groupby(group_columns).apply(pd.DataFrame.to_numpy)))

        n_obs = len(test_array)
        n_columns = len(test.columns)

        test_padded = np.empty((n_obs,n_steps,n_columns), dtype = np.float32)
        for i in range(0,len(test_array)):
            test_padded[i] = np.pad(test_array[i], ((0,n_steps-len(test_array[i])), (0, 0)), 'constant') 
            
        test_padded = test_padded[:, :, len(group_columns):n_columns]
        return test_padded


def binarize_and_split_purchases(data, split=True):
    encoder = MultiLabelBinarizer()
    data1 = pd.concat([data, pd.DataFrame(encoder.fit_transform(data['item_id'].str.split(',')), columns=encoder.classes_)], axis=1)
    data1 = data1.drop(['item_id'], axis=1)
    data1 = data1.groupby(['event_id'], as_index=False).max()
                           
    if split:
        train = data1[data1['valid']==0]
        valid = data1[data1['valid']==1]
        
        train = train.sort_values(by=['event_id'])
        train = train.drop(['valid', 'event_id'], axis=1)
        train = train.values
        
        valid = valid.sort_values(by=['event_id'])
        valid = valid.drop(['event_id', 'valid'], axis=1)
        valid = valid.values
        return train, valid
    else:
        test = data1.sort_values(by=['event_id'])
        test = test.drop(['event_id'], axis=1)
        item_columns = test.columns
        test = test.values
        return test, item_columns
