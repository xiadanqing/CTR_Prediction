from dataprocess import  data
import featureengineer
from featureengineer import persona_features
import pandas as pd
import feather
import gc

appsnum,tags_nums = featureengineer.appsnum,featureengineer.tags_nums

lgbOut_Features = persona_features.main()

afterExpand_df_path = persona_features.afterExpand_df_path

noExpand_dfPath = data.noExpand_dfPath

data = feather.read_dataframe(afterExpand_df_path)
persona_df = pd.read_pickle(noExpand_dfPath)
data = data.merge(persona_df, how='left', on='guid')
del persona_df
gc.collect()

sparse_features = ['app_version', 'guid', 'netmodel', 'newsid', 'geohash', 'ts_hour',
       'device_info','gender']

dense_features=['pos', 'level', 'personidentification', 'followscore', 'personalscore']

var_features = ['applist','new_tag']





