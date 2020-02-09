import dataprocess
import pandas as pd
import collections
from sklearn import preprocessing
import numpy as np

def app_process(k):
    app_path = dataprocess.data_path+'app.csv'
    app = pd.read_csv(app_path)
    app['applist'] = app['applist'].apply(lambda x: str(x)[1:-2])
    app = app.groupby('deviceid')['applist'].apply(lambda x: ' '.join(x)).reset_index()
    app['applist'] = app['applist'].apply(lambda x: x.split(' '))
    applist = app['applist'].tolist()
    applist = [i for p in applist for i in p]
    count = collections.Counter(applist)
    apps = list(count.keys())
    vals = list(count.values())
    max_vals = max(vals)
    weights = [max_vals / val for val in vals]
    weights = preprocessing.minmax_scale(weights)
    app2weight = dict(zip(apps, weights))
    def get_app_weight(app_keys):
        app_values = list(map(lambda x: app2weight[x], app_keys))
        return dict(zip(app_keys, app_values))
    app.loc[app['applist'].isna() == False, 'applist'] = list(
        map(get_app_weight, app.loc[app['applist'].isna() == False, 'applist']))
    app.loc[app['applist'].isna() == False, 'applist'] = app.loc[app['applist'].isna() == False, 'applist'].apply(
        lambda x: dataprocess.topKeys(x, k))
    app.loc[app['applist'].isna() == False, 'applist'] = list(
        map(dataprocess.varFeatEncoder, app.loc[app['applist'].isna() == False, 'applist']))

    apps = app['applist'].tolist()
    apps = [i for p in apps for i in p]
    appsnum = len(set(apps))
    return app,appsnum

if __name__ == '__main__':
    app,appsnum = app_process(5)
    print(np.array(app['applist'].tolist()))