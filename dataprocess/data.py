import math

from keras.preprocessing.sequence import pad_sequences
import dataprocess
import pandas as pd
from datetime import datetime
from dataprocess import app,user
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import numpy as np
import gc

noExpand_dfPath = dataprocess.data_path+'noExpand_df.pkl'
beforeExpand_dfPath = dataprocess.data_path+'beforeExpand_df.feature'
def reduce_mem_usage(df, use_float16=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        if col == 'id':
            df[col] = df[col].astype(str)
            continue
        if is_datetime(df[col]):
            # skip datetime type
            continue
        col_type = df[col].dtype
        if col_type == list:
            continue
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def data_process(k):
    datapath = dataprocess.data_path+'mini_data.csv'
    data = pd.read_csv(datapath)
    data['app_version'] = data['app_version'].apply(lambda x: x.split('.')[:2])
    data['app_version'] = data['app_version'].apply(lambda x: '.'.join(x))
    data['osversion'] = data['osversion'].apply(lambda x: x.split('.')[0])
    data['ts_hour'] = data['ts'].apply(lambda x: datetime.utcfromtimestamp(x // 1000).hour)
    data['new_guid'] = data['guid'].str.cat(data['deviceid'], sep=',', na_rep='-')
    data['guid'] = data['new_guid'].apply(lambda x: dataprocess.catfunc(x))
    data.drop(['new_guid'], axis=1, inplace=True)

    apps, appsnum = dataprocess.app.app_process(k)
    data = data.merge(apps, how='left', on='deviceid')
    del apps
    users, tags_nums = dataprocess.user.user_process(k)
    data = data.merge(users, how='left', on='guid')
    del users
    data['device_info'] = data['device_vendor'].str.cat(data[['device_version', 'osversion']], sep='_', na_rep='-')
    data.drop(['deviceid','timestamp','device_vendor', 'device_version', 'osversion'],
        axis=1, inplace=True)

    data = reduce_mem_usage(data)
    appkeys_list = data['applist'].tolist()
    appkeys_list = pad_sequences(appkeys_list, maxlen=k, padding='post', )
    data['applist'] = appkeys_list.tolist()
    del appkeys_list
    tagkeys_list = data['new_tag'].tolist()
    for i, x in enumerate(tagkeys_list):
        if type(x) != list and math.isnan(x):
            tagkeys_list[i] = [appsnum] * k
    tagkeys_list = pad_sequences(tagkeys_list, maxlen=k, padding='post',value=appsnum )
    data['new_tag'] = tagkeys_list.tolist()
    del tagkeys_list
    noExpand_df = data[
        ['guid', 'applist', 'new_tag', 'personidentification', 'followscore', 'personalscore']].drop_duplicates(
        ['guid']).reset_index(drop=True)

    expand_df = data[['id','guid', 'newsid', 'pos', 'target', 'ts', 'geohash', 'ts_hour', 'device_info', 'app_version', 'gender', 'level',
         'netmodel']]
    del data
    expand_df = expand_df.sort_values('ts').reset_index(drop=True)
    noExpand_df.to_pickle(noExpand_dfPath)
    expand_df.to_feather(beforeExpand_dfPath)
    del noExpand_df, expand_df
    gc.collect()
    return appsnum, tags_nums






