import featureengineer
import category_encoders as ce
from gensim.models import Word2Vec
import dataprocess
from dataprocess import data
import gc
import feather
import numpy as np
import pandas as pd
import lightgbm as lgb

afterExpand_df_path = dataprocess.data_path+'afterExpand_df.feature'

def dense2sparseFeature(df,ruleOut,numTrees):
    columns = df.columns
    lgbIn_features = [x for x in columns if x not in ruleOut]
    newDF = df[lgbIn_features+['target']]
    df.drop(lgbIn_features,axis=1, inplace=True)
    y_preds = lgbPredict(newDF,lgbIn_features,numTrees)
    lgbOut_Features = ['lgbFeatures_'+str(i) for i in range(numTrees)]
    cache_df = pd.DataFrame(data=y_preds,columns=lgbOut_Features)
    df = pd.concat([df,cache_df],axis=1)
    df = data.reduce_mem_usage(df)
    df.to_feather(afterExpand_df_path)
    del cache_df,df,newDF,y_preds
    gc.collect()
    return lgbOut_Features


def lgbPredict(df,lgbIn_features,num_tress):
    num_leaf = 36
    train = df[df.target.isna()==False]
    # 创建成lgb特征的数据集格式
    lgb_train = lgb.Dataset(train[lgbIn_features], train['target'])
    del train
    gc.collect()
    #将参数写成字典下形式
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'binary',  # 目标函数
        'metric': {'auc'},  # 评估函数
        'num_leaves': num_leaf,  # 叶子节点数
        'learning_rate': 0.1,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': -1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

    # 训练 cv and train
    gbm = lgb.train(params, lgb_train, num_boost_round=num_tress)
    del lgb_train
    gc.collect()
    #预测
    y_pred = gbm.predict(df[lgbIn_features],pred_leaf=True)
    del df,gbm
    gc.collect()
    return y_pred

#部分统计特征
def categoryEncoder(dt,col):
    cache = dt[[col,'target']]
    encoder = ce.CatBoostEncoder(cols=col)
    train = cache[cache['target'].isna() == False]
    encoder.fit(train, train['target'])
    cache = encoder.transform(cache)
    dt[col+'_ctr'] = cache[col]
    del train,cache
    gc.collect()
    return dt

#前后x次曝光距当前当前曝光所隔时间（其中后x次曝光为穿越特征在实际线上服务中是无法使用的，这里使用了但没有改，后续可以去掉后x次曝光）
def ts_gap(sort_df,tsGapFeatures):
    added_cols = []
    for f in tsGapFeatures:
        print('------------------ {} ------------------'.format('_'.join(f)))
        tmp = sort_df[f + ['ts']].groupby(f)
        # 前后x次曝光到当前的时间差
        for gap in [1, 2, 3]:
            sort_df['{}_prev{}_exposure_ts_gap'.format('_'.join(f), gap)] = tmp['ts'].shift(0) - tmp['ts'].shift(gap)
            sort_df['{}_next{}_exposure_ts_gap'.format('_'.join(f), gap)] = tmp['ts'].shift(-gap) - tmp['ts'].shift(0)
            added_cols.append('{}_prev{}_exposure_ts_gap'.format('_'.join(f), gap))
            added_cols.append('{}_next{}_exposure_ts_gap'.format('_'.join(f), gap))
        del tmp
        sort_df = data.reduce_mem_usage(sort_df)
    del sort_df['ts']
    gc.collect()
    return sort_df,added_cols

#embedding特征
def emb(df, f1, f2):
    emb_size = 8
    print('====================================== {} {} ======================================'.format(f1, f2))
    tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]
    model = Word2Vec(sentences, size=emb_size, window=5, min_count=5, sg=0, hs=1, seed=2019)
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model:
                vec.append(model[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)
    emb_matrix = np.array(emb_matrix)
    for i in range(emb_size):
        tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]
    del model, emb_matrix, sentences
    tmp = data.reduce_mem_usage(tmp)
    return tmp

def main():
    beforeExpand_dfPath = data.beforeExpand_dfPath
    df = feather.read_dataframe(beforeExpand_dfPath)
    ruleOut = df.columns.tolist()
    # 统计特征
    print('catboostEncoder is begin')
    static_cols = ['guid', 'newsid', 'pos', 'ts_hour', 'netmodel', 'geohash', 'app_version', 'gender', 'level']
    for col in static_cols:
        df = categoryEncoder(df, col)
    # 曝光时间前后特征
    print('timeFeature is begin')
    tsGapFeatures = [
        ['guid'], ['newsid'], ['newsid', 'guid'], ['newsid', 'geohash'],
        ['newsid', 'pos'], ['newsid', 'netmodel']
    ]
    df, added_cols = ts_gap(df, tsGapFeatures)
    # 嵌入特征---测试发现效果无提升，可去掉该嵌入特征
    print('embedding is begin')
    emb_cols = [
        ['guid', 'newsid'],
        ['guid', 'geohash'],
        ['newsid', 'geohash']
    ]
    for f1, f2 in emb_cols:
        df = df.merge(emb(df, f1, f2), on=f1, how='left')
        df = df.merge(emb(df, f2, f1), on=f2, how='left')

    print('dense2sparse by lgb is begin')
    numTrees = 50
    lgbOut_Features = dense2sparseFeature(df, ruleOut, numTrees)
    return lgbOut_Features

