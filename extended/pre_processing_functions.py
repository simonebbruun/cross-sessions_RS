import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from pickle import dump
from pickle import load
from sklearn.preprocessing import MultiLabelBinarizer
from pandasql import sqldf

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


def time(sessions, events, train=True):
    if train:
        time = sqldf('''
        select
        e.event_id,
        e.user_id,
        e.event_time,
        e.valid,
        s.session_id,
        min(s.action_time) as session_start
    
        from events e
         left join sessions s
          on(e.event_id = s.event_id)
        group by
        e.event_id,
        e.user_id,
        e.event_time,
        e.valid,
        s.session_id;
        ''', locals())
        
        time = sqldf('''
        with last_event as(
        select
        t.*,
        max(t.session_start) over(partition by t.event_id) as last_time
        
        from time t)
        , event_number as(
        select
        l.*,
        row_number() over(partition by l.user_id, l.session_start order by l.session_start) as number
        
        from last_event l
        where l.session_start = l.last_time)
        , min_max as(
        select
        t.*,
        min(t.event_time) over(partition by t.user_id, t.session_id, n.number) as min_event_time,
        max(t.event_time) over(partition by t.user_id, t.session_id, n.number) as max_event_time
        
        from time t
         left join event_number n
          on(t.event_id = n.event_id))
        , inter_time as(
        select
        m.event_id,
        m.user_id,
        m.max_event_time,
        m.min_event_time as event_time,
        m.session_id,
        m.session_start,
        m.valid,
        case when (julianday(m.session_start) - lag(julianday(m.session_start)) over(partition by m.user_id order by m.session_start)) > 10 then 1 else 0 end as inter_time
        
        from min_max m
        where event_time = max_event_time)
        , grp as(
        select
        i.*,
        sum(i.inter_time) over(partition by i.user_id order by i.session_start) as grp
        
        from inter_time i)
        select
        g.event_id,
        g.user_id,
        max(g.max_event_time) over(partition by g.user_id, g.grp) as max_event_time,
        g.event_time,
        g.session_id,
        g.session_start,
        g.valid
        
        from grp g
        where julianday(g.session_start) <= julianday(g.event_time);
        ''', locals())
        
    else:
        time = sqldf('''
        select
        e.event_id,
        e.user_id,
        e.event_time,
        s.session_id,
        min(s.action_time) as session_start
    
        from events e
         left join sessions s
          on(e.event_id = s.event_id)
        group by
        e.event_id,
        e.user_id,
        e.event_time,
        s.session_id;
        ''', locals())
        
        time = sqldf('''
        with last_event as(
        select
        t.*,
        max(t.session_start) over(partition by t.event_id) as last_time
        
        from time t)
        , event_number as(
        select
        l.*,
        row_number() over(partition by l.user_id, l.session_start order by l.session_start) as number
        
        from last_event l
        where l.session_start = l.last_time)
        , min_max as(
        select
        t.*,
        min(t.event_time) over(partition by t.user_id, t.session_id, n.number) as min_event_time,
        max(t.event_time) over(partition by t.user_id, t.session_id, n.number) as max_event_time
        
        from time t
         left join event_number n
          on(t.event_id = n.event_id))
        , inter_time as(
        select
        m.event_id,
        m.user_id,
        m.max_event_time,
        m.min_event_time as event_time,
        m.session_id,
        m.session_start,
        case when (julianday(m.session_start) - lag(julianday(m.session_start)) over(partition by m.user_id order by m.session_start)) > 10 then 1 else 0 end as inter_time
        
        from min_max m
        where event_time = max_event_time)
        , grp as(
        select
        i.*,
        sum(i.inter_time) over(partition by i.user_id order by i.session_start) as grp
        
        from inter_time i)
        , purchase_max as(
        select
        g.event_id,
        g.user_id,
        max(g.max_event_time) over(partition by g.user_id, g.grp) as max_event_time,
        g.event_time,
        g.session_id,
        g.session_start
        
        from grp g
        where julianday(g.session_start) <= julianday(g.event_time))
        select
        event_id,
        user_id,
        max_event_time,
        event_time,
        row_number() over(partition by user_id, max_event_time, session_start order by event_time) as event_number,
        session_id,
        session_start
        
        from purchase_max;
        ''', locals())
    
    return time


def sessions_time(sessions, time, train=True):
    if train:
        sessions_time = sqldf('''
        select
        t.user_id,
        t.max_event_time,
        t.valid,
        t.session_start,
        s.action_section,
        s.action_object,
        s.action_type
        
        from time t
         left join sessions s
          on(t.session_id = s.session_id)
        group by
        t.user_id,
        t.max_event_time,
        t.valid,
        t.session_start,
        s.action_section,
        s.action_object,
        s.action_type;
        ''', locals())
    
    else:
        sessions_time = sqldf('''
        select
        t.user_id,
        t.max_event_time,
        t.event_number,
        t.session_start,
        s.action_section,
        s.action_object,
        s.action_type
        
        from time t
         left join sessions s
          on(t.session_id = s.session_id)
        group by
        t.user_id,
        t.max_event_time,
        t.session_start,
        t.event_number,
        s.action_section,
        s.action_object,
        s.action_type;
        ''', locals())
    
    return sessions_time


def events_time(events, time, train=True):
    if train:
        events_time = sqldf('''
        select
        t.user_id,
        t.max_event_time,
        t.event_time,
        t.session_start,
        t.valid,
        e.item_id
        
        from time t
         left join events e
          on(t.user_id = e.user_id and t.event_time = e.event_time);
        ''', locals())
        
    else:
        events_time = sqldf('''
        select
        t.user_id,
        t.max_event_time,
        t.event_time,
        t.session_start,
        t.event_number,
        e.item_id
        
        from time t
         left join events e
          on(t.user_id = e.user_id and t.event_time = e.event_time);
        ''', locals())
    
    return events_time


def compute_time_and_indicator(data):
    encoder = MultiLabelBinarizer()
    data1 = pd.concat([data, pd.DataFrame(encoder.fit_transform(data['item_id'].str.split(',')), columns=encoder.classes_)], axis=1)
    data1 = data1.drop(['item_id'], axis=1)
    data1 = data1.groupby(['event_id', 'session_start'], as_index=False).max()
    data1['event_time'] = pd.to_datetime(data1['event_time'])
    data1['session_start'] = pd.to_datetime(data1['session_start'])
    
    classes = encoder.classes_
    
    data_last = data1.copy(deep=True)
    data_last['diff'] = data_last['event_time']-data_last['session_start']
    data_last.loc[:, 'indicator'] = 0
    data_last.loc[data_last.groupby(['event_id', 'event_time'])['diff'].idxmin(), 'indicator'] = 1
    data_last = data_last.drop([*classes, 'diff', 'event_time'], axis=1)
    data_last = data_last.groupby(['event_id', 'session_start'], as_index=False).max()
    
    for c in classes:
        catagory = data1[data1[c]==1]
        data1_mdl = data1.merge(catagory[['event_id', 'event_time']], how='left', on=['event_id'], suffixes=(None, '_min'))
        data1_mdl['censor_date'] = data1_mdl.groupby('event_id')['session_start'].transform('max')+pd.DateOffset(10)
        data1_mdl[str(c)+'_indicator'] = np.where((data1_mdl['session_start']>data1_mdl['event_time_min']) | (data1_mdl['event_time_min'].isna()),0,1)
        data1_mdl['event_time_min'] = data1_mdl['event_time_min'].fillna(data1_mdl['censor_date'])
        data1_mdl['event_time_min'] = np.where(data1_mdl['session_start']>data1_mdl['event_time_min'],data1_mdl['censor_date'],data1_mdl['event_time_min'])
        time_group_columns = list(data1_mdl.columns)
        time_group_columns = [i for i in time_group_columns if i not in [str(c)+'_indicator', 'event_time_min']]
        data1_mdl = data1_mdl.sort_values('event_time_min').drop_duplicates(time_group_columns)
        data1_mdl[str(c)+'_time'] = (data1_mdl['event_time_min'].dt.date-data1_mdl['session_start'].dt.date).dt.days+1
        data1 = data1_mdl[data1_mdl.columns[~data1_mdl.columns.isin(['event_time_min', 'censor_date'])]]

    data1 = data1.groupby(['event_id', 'session_start'], as_index=False).min()
    data1 = data1.drop(['event_time', *classes], axis=1)
                       
    return data1, data_last


def filter_time(filter, time, train=True):
    if train:
        filter_time = sqldf('''
        select
        t.user_id,
        t.max_event_time,
        t.session_start,
        t.valid,
        e.item_id
        
        from time t
         left join filter e
          on(t.event_id = e.event_id);
        ''', locals())
        
    else:
        filter_time = sqldf('''
        select
        t.user_id,
        t.max_event_time,
        t.session_start,
        t.event_number,
        e.item_id
        
        from time t
         left join filter e
          on(t.event_id = e.event_id);
        ''', locals())
    
    return filter_time


def binarize_filter(data):
    encoder = MultiLabelBinarizer()
    data1 = pd.concat([data, pd.DataFrame(encoder.fit_transform(data['item_id'].str.split(',')), columns=encoder.classes_)], axis=1)
    data1 = data1.drop(['item_id'], axis=1)
    data1 = data1.groupby(['event_id', 'session_start'], as_index=False).max()
                           
    return data1