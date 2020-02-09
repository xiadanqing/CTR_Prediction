import trainmodel
import dataprocess
import featureengineer
from deepctr.models import DeepFM
from deepctr.inputs import  SparseFeat, DenseFeat,VarLenSparseFeat,get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,f1_score
import numpy as np


def train_deepFM():
    k = featureengineer.k
    #缺失值填充+编码处理
    data,appsnum, tags_nums = trainmodel.data,trainmodel.appsnum,trainmodel.tags_nums
    data[trainmodel.sparse_features] = data[trainmodel.sparse_features].fillna('-1', )
    for feat in trainmodel.dense_features:
        data[feat].fillna(data[feat].dropna().mean(), inplace=True)

    for feat in trainmodel.sparse_features:
        data[feat] = data[feat].apply(lambda x:str(x))
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[trainmodel.dense_features] = mms.fit_transform(data[trainmodel.dense_features])


    #数据格式转换
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max()+1, embedding_dim=8)
                              for i, feat in enumerate(trainmodel.sparse_features)] + \
                             [DenseFeat(feat, 1, ) for feat in trainmodel.dense_features]

    lgbOut_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max()+1, embedding_dim=1)
                              for i, feat in enumerate(trainmodel.lgbOut_Features)]

    key2index_len = {'applist': appsnum+1, 'new_tag': tags_nums}
    varlen_features = [VarLenSparseFeat('%s' % i, vocabulary_size=key2index_len[i], maxlen=k, embedding_dim=8, combiner='mean',weight_name=None) for i
                       in trainmodel.var_features]

    dnn_feature_columns = fixlen_feature_columns + varlen_features
    linear_feature_columns = fixlen_feature_columns + varlen_features + lgbOut_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    sparse_dense_features = trainmodel.sparse_features + trainmodel.dense_features + trainmodel.lgbOut_Features

    train, test = train_test_split(data, test_size=0.2)


    train_model_input = {name: train[name] for name in sparse_dense_features}
    test_model_input = {name: test[name] for name in sparse_dense_features}
    for x in trainmodel.var_features:
        if x == 'applist':
            train_model_input[x] = np.array(train[x].tolist())
            test_model_input[x] = np.array(test[x].tolist())
        if x == 'new_tag':
            train_model_input[x] = np.array(train[x].tolist())-appsnum
            test_model_input[x] = np.array(test[x].tolist())-appsnum
    # 模型
    model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   dnn_hidden_units=(50, 30, 30), l2_reg_linear=0.001, l2_reg_embedding=0.001,
                   l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0.1, dnn_activation='relu', dnn_use_bn=True,
                   task='binary')
    model.compile("adam", "binary_crossentropy",metrics=['AUC'], )

    history = model.fit(train_model_input, train['target'].values,
                        batch_size=256, epochs=1, verbose=2, validation_split=0.2, )

    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test AUC", round(roc_auc_score(test['target'].values, pred_ans), 4))

if __name__ == "__main__":
    train_deepFM()


