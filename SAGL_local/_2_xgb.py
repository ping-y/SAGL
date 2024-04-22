## Yunchuan Kong
## 2019 Copyright Reserved

from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import sys
from sklearn import metrics
import pickle
from sklearn.preprocessing import StandardScaler
# import xgboost as xgb
import time


def get_feature_graph_xgb(pos_weight,expression_train,y_train):
    xgb = XGBClassifier(scale_pos_weight=pos_weight, random_state=42, objective='binary:logistic',
                              use_label_encoder=False)

    def recode_feature_name_array(f_nparray):
        def recode_feature_name(name_str):
            if name_str[0] is "f":
                return int(name_str[1:])
            else:
                return int(-2)

        return list(map(recode_feature_name, f_nparray))

    def parse_booster(tree_df):
        f_idx = recode_feature_name_array(np.array(tree_df['Feature']))
        # print('f_idx', f_idx)  # f_idx [1, 93, 476, -2, -2, 28, 84, -2, -2, -2, -2]
        nodes = np.array(tree_df['Node'])
        # print('nodes', nodes)  # nodes [ 0  1  2  3  4  5  6  7  8  9 10]
        roots = np.array(tree_df.loc[tree_df['Feature'] != 'Leaf', 'Node'])
        # print('roots',roots)   # roots [0 1 2 5 6]
        right = np.array(nodes[2::2])
        # print('right', right)  # [ 2  4  6  8 10]
        left = np.array(nodes[1::2])
        # print('left', left)  # [1 3 5 7 9]
        edge_list = np.stack((roots, left, roots, right), axis=1)
        # print('1. edge_list', edge_list)
        # print('np.take(f_idx, edge_list)', np.take(f_idx, edge_list))
        edge_list = np.reshape(np.take(f_idx, edge_list), [-1, 2])
        # print('2. edge_list', edge_list)
        edge_list = edge_list[edge_list.min(axis=1) >= 0, :]
        # print('3. edge_list', edge_list)
        # 3. edge_list
        # [[ 49 199]
        #  [ 49   4]
        #  [  4   1]]
        return edge_list


    # xgb = XGBClassifier(n_estimators=n_trees, n_jobs=-1)
    xgb.fit(expression_train, y_train)

    f_importance = xgb.feature_importances_

    forest = xgb.get_booster().trees_to_dataframe()
    # print('1. forest', forest)

    feat_split=forest[['Feature','Split']].drop_duplicates()
    feat_split=feat_split[feat_split['Feature']!= 'Leaf']

    forest = list(forest.groupby(by='Tree'))
    # print('2. forest', forest)
    forest = list(list(zip(*forest))[1])
    # print('3. forest', forest)

    elist = np.unique(np.vstack(list(map(parse_booster, forest))), axis=0)


    return elist, feat_split    # (边数, 2)


def get_feature_graph_rf(pos_weight,expression_train,y_train):
    # rf = RandomForestClassifier(n_estimators=n_trees, bootstrap=False,  n_jobs=-1)
    rf = RandomForestClassifier(random_state=42, verbose=1, bootstrap=True, class_weight="balanced",
                                   n_estimators=150, max_depth=5, min_samples_split=10)
    rf.fit(expression_train, y_train)

    ## feature importance from the forest
    f_importance = rf.feature_importances_

    def parse_tree(decision_tree):
        tree = decision_tree.tree_
        parse_list = np.array(list(zip(tree.feature, tree.children_left, tree.children_right)))
        roots = np.array(range(np.shape(parse_list)[0]))
        edge_list = np.stack((roots, parse_list[:, 1], roots, parse_list[:, 2]), axis=1)
        edge_list = np.reshape(np.take(parse_list[:, 0], edge_list), [-1, 2])
        edge_list = edge_list[edge_list.min(axis=1) >= 0, :]
        return edge_list

    elist = np.unique(np.vstack(list(map(parse_tree, rf.estimators_))), axis=0)
    return elist    # (边数, 2)


def focal_loss(pred, y, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     pred: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     y: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    zeros = tf.zeros_like(pred, dtype=pred.dtype)

    # For positive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
    pos_p_sub = tf.where(y > zeros, y - pred, zeros)  # positive sample 寻找正样本，并进行填充

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = tf.where(y > zeros, zeros, pred)  # negative sample 寻找负样本，并进行填充
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))

    return tf.reduce_sum(per_entry_cross_ent)



def get_feat_split():
    for has_CIs in [False]:

        if has_CIs:
            ci_flag = 'wtCI'
        else:
            ci_flag = 'woCI'

        ls_rsts_=[]

        for seed in range(10):

            print('=====forgeNet    round    %s    seed%d    ======='%(ci_flag,seed))

            trainIDs_ls = pickle.load(open(dir+'/GCT2023_data/data/trainIDs_ls_seed%d_%s.pkl'%(seed,ci_flag), "rb"))
            valIDs_ls = pickle.load(open(dir+'/GCT2023_data/data/valIDs_ls_seed%d_%s.pkl'%(seed,ci_flag), "rb"))
            testIDs_ls = pickle.load(open(dir+'/GCT2023_data/data/testIDs_ls_seed%d_%s.pkl'%(seed,ci_flag), "rb"))

            dic_train=pickle.load(open(dir+'/GCT2023_data/data/dic_train_forgeNet_seed%d_%s.pkl'%(seed,ci_flag), 'rb'))
            dic_test = pickle.load(open(dir+'/GCT2023_data/data/dic_test_forgeNet_seed%d_%s.pkl'%(seed,ci_flag), 'rb'))

            # dic_test:
            # {0:   patient_id: [array([1550, 2053, 2735,.....
            #         feat_df: [236 rows x 75 columns]    -   columns : ['年龄', 'BMI', '肥胖', '贫血', '腹水', ... ,  '性别0', '性别1', '婚姻0', ...]
            #         label:  array([0, 0, 0, 0, 1,

            print(len(dic_train[0]))  # 3
            print(dic_train[0][1].shape)
            feat_num=dic_train[0][1].shape[-1]

            # # 标准化基本特征
            # for p in list(dic_train.keys()):  # [train_ids: np.array, train_df: df, train_y: np.array]
            #     df_basic_feats = dic_train[p][1]
            #     df_basic_feats_test = dic_test[p][1]
            #
            #     scaler = StandardScaler()
            #     scaler.fit(df_basic_feats)
            #     df_basic_feats = scaler.transform(df_basic_feats)
            #     df_basic_feats_test = scaler.transform(df_basic_feats_test)
            #
            #     dic_train[p][1] = pd.DataFrame(df_basic_feats)
            #     dic_test[p][1] = pd.DataFrame(df_basic_feats_test)

            ls_rst=[]
            for cv_i in range(10):
                train_val_samples=dic_train[cv_i][1]
                # print('train_val_samples',train_val_samples)

                train_index = np.zeros_like(trainIDs_ls[cv_i])
                for i, aa in np.ndenumerate(trainIDs_ls[cv_i]):
                    train_index[i] = np.where(dic_train[cv_i][0] == aa)[0]
                print('train_index',train_index.shape,train_index)

                val_index = np.zeros_like(valIDs_ls[cv_i])
                for i, aa in np.ndenumerate(valIDs_ls[cv_i]):
                    val_index[i] = np.where(dic_train[cv_i][0] == aa)[0]
                print('val_index', val_index.shape)

                # train_index = train_val_samples[train_val_samples['patient_id'].isin(trainIDs_ls[cv_i])].index
                # val_index = train_val_samples[train_val_samples['patient_id'].isin(valIDs_ls[cv_i])].index
                if set(list(train_index))&set(list(val_index)) or len(set(list(train_index))|set(list(val_index)))!=train_val_samples.shape[0]:
                    raise Exception('数据划分有问题')

                # dic_featName2cols=dict(zip(list(range(train_val_samples.shape[1])),list(train_val_samples.columns)))
                # print('dic_featName2cols',dic_featName2cols)
                expression_train=train_val_samples.loc[train_index].values
                y_train=dic_train[cv_i][2][train_index]
                expression_val = train_val_samples.loc[val_index].values
                y_val = dic_train[cv_i][2][val_index]
                expression_test=dic_test[cv_i][1].values
                y_test=dic_test[cv_i][2]

                pos_weight = (y_train.shape[0] - sum(y_train)) / sum(y_train)
                elist, feat_split=get_feature_graph_xgb(pos_weight, expression_train,y_train)
                # elist = get_feature_graph_rf(pos_weight, expression_train, y_train)
                print('elist.shape',elist.shape,elist)   # (781, 2)
                    #   [[ 0  0]
                    #  [ 0  1]
                    #  [ 0  2]
                    #  ...
                    #  [72 64]
                    #  [72 65]
                    #  [72 70]]

                dic_featName2cols = dict(zip(map(str,list(range(train_val_samples.shape[1]))), list(train_val_samples.columns)))
                print('dic_featName2cols', dic_featName2cols)

                feat_split['feat_name']=feat_split['Feature'].apply(lambda x: dic_featName2cols[x[1:]])
                feat_split['Split'].astype(float)
                feat_split.to_csv(dir+'/GCT2023_data/data_xgb_GCT/xgb_split_pos___%s_seed%d_cv%d.csv'%(ci_flag, seed,cv_i))


                # 取左闭右开区间 ，-inf -- +inf
                dic=feat_split.groupby('feat_name').apply(get_sorted_split_value)
                dic=dict(zip(dic.index, dic.values))

                for di in dic:
                    print(di,dic[di])   # {(-inf, 0.5): 0, (0.5, inf): 1}
                # pickle.dump(dic,open('F:/GCT2023_data/data_xgb_GCT/dic_split_interval___%s_seed%d_cv%d.pkl'%(ci_flag, seed,cv_i),'wb'))



def get_sorted_split_value(df_group):
    sorted_nda=df_group['Split'].sort_values().values
    # print('sorted_nda',sorted_nda.shape)
    dic=dict()

    c=0
    dic[(-float('inf'), sorted_nda[0])]=c
    for i in range(1,sorted_nda.shape[0]):
        c+=1
        dic[(sorted_nda[i-1], sorted_nda[i])]=c
    dic[(sorted_nda[-1], float('inf'))]=c+1
    return dic


def get_bucket():
    """"""

    def get_bucket_id(df_group, dic_split_interval,discrete_feats):
        ls = []
        for c in dic_split_interval:   # c是特征名称
            cur = df_group.iloc[0][c]

            if (c in discrete_feats and cur==1) or c not in discrete_feats:
                for k in dic_split_interval[c]:
                    if cur > k[0] and cur <= k[1]:
                        ls.append(c + "_" + str(k))
                        break
        return ls


    for has_CIs in [False]:
        ci_flag = 'wtCI' if has_CIs else 'woCI'

        for seed in range(10):
            print('=====forgeNet    round    %s    seed%d    ======='%(ci_flag,seed))
            dic_train=pickle.load(open(dir+'/GCT2023_data/data/dic_train_forgeNet_seed%d_%s.pkl'%(seed,ci_flag), 'rb'))
            dic_test = pickle.load(open(dir+'/GCT2023_data/data/dic_test_forgeNet_seed%d_%s.pkl'%(seed,ci_flag), 'rb'))

            # dic_test:
            # {0:   patient_id: [array([1550, 2053, 2735,.....
            #         feat_df: [236 rows x 75 columns]    -   columns : ['年龄', 'BMI', '肥胖', '贫血', '腹水', ... ,  '性别0', '性别1', '婚姻0', ...]
            #         label:  array([0, 0, 0, 0, 1,

            # print(list(dic_test[0][1].columns))
            discrete_feats=['性别0', '性别1', '婚姻0', '婚姻1', '肿瘤部位0', '肿瘤部位1', '肿瘤形态0', '肿瘤形态1', '肿瘤形态2', '肿瘤形态3', '肿瘤形态4',
                         '肿瘤性质1', '肿瘤性质2', '肿瘤性质3', '肿瘤性质6', '肿瘤性质7', '梗阻0', '梗阻1', '梗阻2', '梗阻3', '梗阻4', '梗阻5', '梗阻6',
                         '套叠0', '套叠1', '套叠2', '穿孔0', '穿孔1', '穿孔2', '穿孔3',
                         'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'N0', 'N1', 'N2', 'N3', 'N4', 'M0', 'M1', 'M2', 'M3', 'M4', 'M5',
                         '手术性质0', '手术性质1', '手术性质2', '手术性质3', '造口0', '造口1', '造口2', '造口3', '造口5', '扩大0', '扩大1', '新辅类型0', '新辅类型1']

            print(len(dic_train[0]))  # 3
            print(dic_train[0][1].shape)

            ls_rst=[]
            for cv_i in range(10):
                dic_split_interval=pickle.load(open(dir+'/GCT2023_data/data_xgb_GCT/dic_split_interval___%s_seed%d_cv%d.pkl' % (ci_flag, seed, cv_i), 'rb'))

                df_train=pd.DataFrame(dic_train[cv_i][0],columns=['patient_id'])
                df_train=pd.concat([df_train, dic_train[cv_i][1].reset_index(drop=True)],axis=1)

                df_test = pd.DataFrame(dic_test[cv_i][0], columns=['patient_id'])
                df_test = pd.concat([df_test, dic_test[cv_i][1].reset_index(drop=True)], axis=1)

                df = pd.concat([df_train, df_test], axis=0)


                tmp=df.groupby('patient_id').apply(get_bucket_id, dic_split_interval,discrete_feats)
                tmp=pd.DataFrame(zip(tmp.index, tmp.values),columns=['patient_id','cx_bucket'])
                print('tmp',tmp)
                # tmp['cx_feat_num']=tmp['cx_bucket'].apply(lambda x:len(x))
                # print(tmp['cx_feat_num'].describe())
                pickle.dump(tmp, open(dir+'/GCT2023_data/data_xgb_GCT/pID_cxBucket___%s_seed%d_cv%d_v2.pkl' % (ci_flag, seed, cv_i),  'wb'))
                # tmp.to_csv('F:/GCT2023_data/data_xgb_GCT/pID_cxBucket___%s_seed%d_cv%d.csv' % (ci_flag, seed, cv_i))



def map_bucket_to_ids():
    process=[1,3]

    if 1 in process:
        # 统计个体最多有多少个分桶特征，即矩阵中cx占几列
        for has_CIs in [False]:
            ci_flag = 'wtCI' if has_CIs else 'woCI'
            ls_tmp=[]
            for seed in range(10):
                for cv_i in range(10):
                    bucket_cx = pickle.load(open(dir+'/GCT2023_data/data_xgb_GCT/pID_cxBucket___%s_seed%d_cv%d_v2.pkl' % (ci_flag, seed, cv_i), 'rb'))
                    # print(bucket_cx)
                    bucket_cx['cx_bucket_num']=bucket_cx['cx_bucket'].apply(lambda x: len(x))
                    print("每个人的分桶特征数量最大值：",bucket_cx['cx_bucket_num'].max())
                    ls_tmp.append(bucket_cx['cx_bucket_num'].max())

                    pickle.dump(bucket_cx['cx_bucket_num'].max(), open(dir+'/GCT2023_data/data_xgb_GCT/max_cxBucket_num___%s_seed%d_cv%d_v2.pkl' % (ci_flag, seed, cv_i), 'wb'))

            ls_tmp=pd.DataFrame(ls_tmp)
            print(ls_tmp.describe())    # woCI: 58(min)-61-62-62-65(max);  wtCI: 59(min)-62-64-65-68(max);


    if 2 in process:
        # 统计整个数据集上，分桶特征总数
        for has_CIs in [False]:
            ci_flag = 'wtCI' if has_CIs else 'woCI'
            ls_tmp=[]
            for seed in range(1):
                for cv_i in range(1):
                    bucket_cx = pickle.load(open(dir+'/GCT2023_data/data_xgb_GCT/pID_cxBucket___%s_seed%d_cv%d_v2.pkl' % (ci_flag, seed, cv_i), 'rb'))
                    # print(bucket_cx)
                    set_=set()
                    for i in bucket_cx['cx_bucket']:
                        set_|=set(i)
                    print('分桶特征总数：',len(set_))
                    ls_tmp.append(len(set_))
            ls_tmp = pd.DataFrame(ls_tmp)
            print(ls_tmp.describe())    # woCI: 646(min)-672-686-694-727(max);  wtCI: 662(min)-698-711-721-765(max);


    if 3 in process:
        # 映射到数字，注意每个数据集上对应的分桶特征总数是不同的，因此要分别处理并保存
        for has_CIs in [False]:
            ci_flag = 'wtCI' if has_CIs else 'woCI'
            # ls_tmp=[]
            for seed in range(10):
                for cv_i in range(10):
                    bucket_cx = pickle.load(open(dir+'/GCT2023_data/data_xgb_GCT/pID_cxBucket___%s_seed%d_cv%d_v2.pkl' % (ci_flag, seed, cv_i), 'rb'))
                    # print(bucket_cx)
                    set_=set()
                    for i in bucket_cx['cx_bucket']:
                        set_|=set(i)
                    print('分桶特征总数：',len(set_))
                    dic=dict(zip(sorted(list(set_)),list(range(len(set_)))))
                    bucket_cx['cx_bucket_idx']=bucket_cx['cx_bucket'].apply(lambda x: [dic[i] for i in x])
                    # bucket_cx.to_csv('F:/GCT2023_data/tmp/pID_cxBucket_cxIdx.csv')
                    pickle.dump(bucket_cx, open(dir+'/GCT2023_data/data_xgb_GCT/pID_cxBucket_cxIdx___%s_seed%d_cv%d_v2.pkl' % (ci_flag, seed, cv_i),'wb'))
                    pickle.dump(dic, open(dir+'/GCT2023_data/data_xgb_GCT/dic_cxBucket2Idx___%s_seed%d_cv%d_v2.pkl' % (ci_flag, seed, cv_i), 'wb'))


def get_imputed_ids_and_masks():
    for has_CIs in [False]:
        ci_flag = 'wtCI' if has_CIs else 'woCI'
        # ls_tmp=[]
        for seed in range(10):
            for cv_i in range(10):
                pID_cxBucket_cxIdx = pickle.load(open(dir+'/GCT2023_data/data_xgb_GCT/pID_cxBucket_cxIdx___%s_seed%d_cv%d_v2.pkl' % (ci_flag, seed, cv_i),  'rb'))
                dic_cxBucket2Idx = pickle.load(open(dir+'/GCT2023_data/data_xgb_GCT/dic_cxBucket2Idx___%s_seed%d_cv%d_v2.pkl' % (ci_flag, seed, cv_i), 'rb'))
                max_cxBucket_num = pickle.load(open(dir+'/GCT2023_data/data_xgb_GCT/max_cxBucket_num___%s_seed%d_cv%d_v2.pkl' % (ci_flag, seed, cv_i), 'rb'))



                pID_cxBucket_cxIdx['ls_cx_ints_imputed'] = pID_cxBucket_cxIdx['cx_bucket_idx'].apply(lambda x: impute_cx_ints(x, max_cxBucket_num, len(dic_cxBucket2Idx)))
                pID_cxBucket_cxIdx['cx_mask'] = pID_cxBucket_cxIdx['cx_bucket_idx'].apply(lambda x: get_mask_dx_ints(x, max_cxBucket_num))
                print(pID_cxBucket_cxIdx[['ls_cx_ints_imputed', 'cx_mask']])
                # pID_cxBucket_cxIdx.to_csv('F:/GCT2023_data/tmp/pID_cxBucket_cxIdx.csv')
                pickle.dump(pID_cxBucket_cxIdx, open( dir+'/GCT2023_data/data_xgb_GCT/pID_cxBucket_cxIdx___%s_seed%d_cv%d_v2.pkl' % (ci_flag, seed, cv_i),  'wb'))


def impute_cx_ints(x, max_num_codes, len_diseases):
    new_x=x.copy()
    imputed=[len_diseases]*(max_num_codes-len(x))
    new_x.extend(imputed)
    return new_x


def impute_dx_ints(x, max_num_codes, len_diseases):
    new_xs=[]

    for i in [0,1]:
        new_x=x[i].copy()
        imputed=[len_diseases]*(max_num_codes-len(x[i]))
        new_x.extend(imputed)
        new_xs.append(new_x)
    return new_xs


def get_mask_dx_ints(x, max_num_codes):
    mask=[1]*len(x)
    imputed=[0]*(max_num_codes-len(x))
    mask.extend(imputed)
    return mask


def get_edge_indice_value(x, dic_edge2RR):
    indices=[]
    values=[]
    x = list(x)
    for i in range(len(x)-1):
        for j in range(i+1,len(x)):
            if (x[i], x[j]) in dic_edge2RR:
                indices.append((i, j))
                values.append(dic_edge2RR[(x[i],x[j])])
            if (x[j],x[i]) in dic_edge2RR:
                indices.append((j, i))
                values.append(dic_edge2RR[(x[j], x[i])])
    return [indices,values]


def get_dic_edge2conditionP(pID_cxBucket_cxIdx,col_dim):
    ls=[]
    for cx in pID_cxBucket_cxIdx['cx_bucket_idx']:
        ini=np.zeros(col_dim)
        ini[cx]=1
        ls.append(ini)
    ls=np.stack(ls,axis=0)
    # print('ls',ls)
    # P(A/B)  -> (A, B)
    sum_=ls.sum(axis=0)
    # print(sum_.shape)     # (669,)

    co_ocurrence=np.matmul(ls.T,ls)

    dic_edge2conditionP=dict()
    for i in range(col_dim-1):
        for j in range(i+1, col_dim):
            co_occur=co_ocurrence[i][j]
            if co_occur:
                dic_edge2conditionP[(i,j)]=co_occur/sum_[j]
                dic_edge2conditionP[(j,i)] = co_occur / sum_[i]

    # print('dic_edge2conditionP',dic_edge2conditionP)
    return dic_edge2conditionP


def get_cx_prior_guide():
    # 得到cx的 'dx_prior_indices', 'dx_prior_values'
    for has_CIs in [False]:
        ci_flag = 'wtCI' if has_CIs else 'woCI'
        # ls_tmp=[]
        for seed in range(10):
            for cv_i in range(10):
                pID_cxBucket_cxIdx = pickle.load(open(dir+'/GCT2023_data/data_xgb_GCT/pID_cxBucket_cxIdx___%s_seed%d_cv%d_v2.pkl' % (ci_flag, seed, cv_i), 'rb'))
                dic_cxBucket2Idx = pickle.load(open(dir+'/GCT2023_data/data_xgb_GCT/dic_cxBucket2Idx___%s_seed%d_cv%d_v2.pkl' % (ci_flag, seed, cv_i), 'rb'))
                max_cxBucket_num = pickle.load(open(dir+'/GCT2023_data/data_xgb_GCT/max_cxBucket_num___%s_seed%d_cv%d_v2.pkl' % (ci_flag, seed, cv_i), 'rb'))
                # print(pID_cxBucket_cxIdx[['cx_bucket', 'cx_bucket_idx']])
                # print(np.array(list(pID_cxBucket_cxIdx[ 'cx_bucket_idx'])).max(),np.array(list(pID_cxBucket_cxIdx[ 'cx_bucket_idx'])).min())    # 668, 0

                trainIDs_ls = pickle.load(open(dir+'/GCT2023_data/data/trainIDs_ls_seed%d_%s.pkl' % (seed, ci_flag), "rb"))[cv_i]
                valIDs_ls = pickle.load(open(dir+'/GCT2023_data/data/valIDs_ls_seed%d_%s.pkl' % (seed, ci_flag), "rb"))[cv_i]
                # testIDs_ls = pickle.load(open(dir+'/GCT2023_data/data/testIDs_ls_seed%d_%s.pkl' % (seed, ci_flag), "rb"))[cv_i]

                has_val=True
                if has_val:
                    trainIDs_ls.extend(valIDs_ls)   # 把val set纳入统计
                trainIDs_ls=pd.DataFrame(trainIDs_ls, columns=['patient_id'])
                stat_pID_cxBucket_cxIdx=pd.merge(trainIDs_ls, pID_cxBucket_cxIdx, on=['patient_id'])
                dic_edge2conditionP=get_dic_edge2conditionP(stat_pID_cxBucket_cxIdx,len(dic_cxBucket2Idx))

                pID_cxBucket_cxIdx['prior'] = pID_cxBucket_cxIdx['cx_bucket_idx'].apply(lambda x: get_edge_indice_value(x, dic_edge2conditionP))
                pID_cxBucket_cxIdx['cx_prior_indices'] = pID_cxBucket_cxIdx['prior'].apply(lambda x: x[0])
                pID_cxBucket_cxIdx['cx_prior_values'] = pID_cxBucket_cxIdx['prior'].apply(lambda x: x[1])
                del pID_cxBucket_cxIdx['prior']
                # print(pID_cxBucket_cxIdx['cx_prior_indices'])
                # print(pID_cxBucket_cxIdx['cx_prior_values'])
                # print(pID_cxBucket_cxIdx['cx_prior_values'].apply(lambda x: len(x)).describe())
                print('pID_cxBucket_cxIdx.columns',list(pID_cxBucket_cxIdx.columns))
                pickle.dump(pID_cxBucket_cxIdx, open(dir+'/GCT2023_data/data_xgb_GCT/pID_cxBucket_cxIdx___%s_seed%d_cv%d_v2.pkl' % (ci_flag, seed, cv_i), 'wb'))


def get_feat_outcome_asso():
    for has_CIs in [False]:
        ci_flag = 'wtCI' if has_CIs else 'woCI'
        # ls_tmp=[]
        for seed in range(10):
            for cv_i in range(10):
                pID_cxBucket_cxIdx = pickle.load(open(dir+'/GCT2023_data/data_xgb_GCT/pID_cxBucket_cxIdx___%s_seed%d_cv%d.pkl' % (ci_flag, seed, cv_i), 'rb'))
                dic_cxBucket2Idx = pickle.load(open(dir+'/GCT2023_data/data_xgb_GCT/dic_cxBucket2Idx___%s_seed%d_cv%d.pkl' % (ci_flag, seed, cv_i), 'rb'))
                max_cxBucket_num = pickle.load(open(dir+'/GCT2023_data/data_xgb_GCT/max_cxBucket_num___%s_seed%d_cv%d.pkl' % (ci_flag, seed, cv_i), 'rb'))

                dic_train = pickle.load(open(dir+'/GCT2023_data/data/dic_train_seed%d_%s.pkl' % (seed, ci_flag), 'rb'))[cv_i]
                dic_test = pickle.load(open(dir+'/GCT2023_data/data/dic_test_seed%d_%s.pkl' % (seed, ci_flag), 'rb'))[cv_i]
                #   dic_train       :  [train_ids: np.array, basic_feats: ndarray, train_y: np.array, df_disease_p]
                #   df_disease_p:   ['patient_id', 'ls_dx_ints', 'ls_dx_names', 'dx_prior_indices', 'dx_prior_values']
                dis2no = pickle.load(open(dir+'/GCT2023_data/data/dis2no.pkl', "rb"))
                len_diseases = len(dis2no)

                # cx
                pID_y=pd.DataFrame(zip(dic_train[0],dic_train[2]),columns=['patient_id', 'y'])
                pID_y=pd.merge(pID_y,pID_cxBucket_cxIdx,on='patient_id')
                dic_feat_outcome=get_dic_feat_outcome(pID_y[['cx_bucket_idx','y']],'cx_bucket_idx',len(dic_cxBucket2Idx))

                # dx
                pID_y = pd.DataFrame(zip(dic_train[0], dic_train[2]), columns=['patient_id', 'y'])
                pID_y=pd.merge(dic_train[-1][['patient_id','ls_dx_ints']],pID_y,on='patient_id')
                dic_dis_outcome=get_dic_feat_outcome(pID_y[['ls_dx_ints','y']], 'ls_dx_ints', len_diseases)

                # 构建cx_oc和dx_oc
                pID_cxBucket_cxIdx['cx_oc'] = pID_cxBucket_cxIdx['cx_bucket_idx'].\
                    apply(lambda x: [[dic_feat_outcome[('death',i)] for i in x],[dic_feat_outcome[(i,'death')] for i in x]])

                tmp=pd.concat([dic_train[-1][['patient_id', 'ls_dx_ints']],dic_test[-1][['patient_id','ls_dx_ints']]],axis=0)
                pID_cxBucket_cxIdx=pd.merge(pID_cxBucket_cxIdx, tmp, on='patient_id')
                pID_cxBucket_cxIdx['dx_oc'] =pID_cxBucket_cxIdx['ls_dx_ints'].\
                    apply(lambda x: [[dic_dis_outcome[('death',i)] for i in x],[dic_dis_outcome[(i,'death')] for i in x]])
                del pID_cxBucket_cxIdx['ls_dx_ints']

                # 补齐cx_oc和dx_oc
                max_num_codes_dx=30
                pID_cxBucket_cxIdx['cx_oc_imputed']=pID_cxBucket_cxIdx['cx_oc'].apply( lambda x: impute_dx_ints(x, max_cxBucket_num, 0))
                pID_cxBucket_cxIdx['dx_oc_imputed'] = pID_cxBucket_cxIdx['dx_oc'].apply(lambda x: impute_dx_ints(x, max_num_codes_dx, 0))

                print('pID_cxBucket_cxIdx.shape',pID_cxBucket_cxIdx.shape)
                pickle.dump(pID_cxBucket_cxIdx, open(dir+'/GCT2023_data/data_xgb_GCT/pID_cxBucket_cxIdx___%s_seed%d_cv%d.pkl' % (ci_flag, seed, cv_i), 'wb'))


def get_dic_feat_outcome(df,col, col_dim):
    y=df['y'].values
    sum_y=y.sum()
    ls = []
    for cx in df[col]:
        ini = np.zeros(col_dim)
        ini[cx] = 1
        ls.append(ini)
    ls = np.stack(ls, axis=0)
    sum_ = ls.sum(axis=0)

    dic_feat_outcome=dict()
    for i in range(ls.shape[1]):
        co_occur=sum(ls[:,i]+y==2)
        if co_occur:
            dic_feat_outcome[(i,'death')]=co_occur/sum_y
            dic_feat_outcome[('death',i)] = co_occur / sum_[i]
        else:
            dic_feat_outcome[(i, 'death')] =0
            dic_feat_outcome[('death', i)] =0
    return dic_feat_outcome


def get_dx_cx_condiP(pid_dx_cx_train, len_cx, len_diseases):
    # pid_dx_cx_train: 'patient_id', 'ls_dx_ints', 'cx_bucket_idx'
    print(pid_dx_cx_train['ls_dx_ints'])
    print(pid_dx_cx_train['cx_bucket_idx'])

    ls_dx_ints=list(pid_dx_cx_train['ls_dx_ints'])
    cx_bucket_idx=list(pid_dx_cx_train['cx_bucket_idx'])

    new_cx_bucket_idx=[list(np.array(i)+len_diseases) for i in cx_bucket_idx]

    cx_dx_ls=[ls_dx_ints[i]+new_cx_bucket_idx[i] for i in range(len(ls_dx_ints))]
    print(len(cx_dx_ls))

    # 构建矩阵
    ls = []
    for idx in cx_dx_ls:
        ini = np.zeros( len_cx+len_diseases)
        ini[idx] = 1
        ls.append(ini)
    ls = np.stack(ls, axis=0)
    print('ls',ls.shape,ls)
    # P(A/B)  -> (A, B)
    sum_ = ls.sum(axis=0)
    print(sum_.shape)     # (989,)

    co_ocurrence = np.matmul(ls.T, ls)

    condP_mtx=co_ocurrence/sum_    # 除0会产生nan
    condP_mtx = np.nan_to_num(condP_mtx)
    print(condP_mtx.shape)

    dx_cx_mtx=condP_mtx[:len_diseases, len_diseases:]   # (dx&cx)/cx
    cx_dx_mtx=condP_mtx[len_diseases:, :len_diseases]   # (dx&cx)/dx

    return dx_cx_mtx, cx_dx_mtx


def get_p_dxcx_condiP(pid_dx_cx_train, dx_cx_mtx, cx_dx_mtx, max_code_dx, max_code_cx):
    # 构建每个患者的dx_cx_condiP矩阵和cx_dx_condiP矩阵

    # pid_dx_cx_train: 'patient_id', 'ls_dx_ints', 'cx_bucket_idx'
    # dx_cx_condiP矩阵:(len_dx, len_cx)
    # cx_dx_mtx矩阵:(len_cx, len_dx)
    def get_row_dxcx_condiP(ls_dx_ints, cx_bucket_idx, dx_cx_mtx, cx_dx_mtx, max_code_dx, max_code_cx):
        tmp=dx_cx_mtx[ls_dx_ints,:]
        p_dx_cx=tmp[:, cx_bucket_idx]

        tmp=cx_dx_mtx[cx_bucket_idx,:]
        p_cx_dx=tmp[:, ls_dx_ints]

        # 补齐
        p_dx_cx_padded=np.zeros([max_code_dx, max_code_cx])
        p_dx_cx_padded[:p_dx_cx.shape[0],:p_dx_cx.shape[1]]=p_dx_cx

        p_cx_dx_padded=np.zeros([max_code_cx, max_code_dx])
        p_cx_dx_padded[:p_cx_dx.shape[0],:p_cx_dx.shape[1]]=p_cx_dx

        return (p_dx_cx_padded, p_cx_dx_padded)


    pid_dx_cx_train['dxcx_condiP']=pid_dx_cx_train.apply(lambda row:
                                                             get_row_dxcx_condiP(row['ls_dx_ints'], row['cx_bucket_idx'], dx_cx_mtx, cx_dx_mtx, max_code_dx, max_code_cx), axis=1)
    print(pid_dx_cx_train['dxcx_condiP'])
    print(pid_dx_cx_train['dxcx_condiP'].values[0][0].shape, pid_dx_cx_train['dxcx_condiP'].values[0][0])
    print(pid_dx_cx_train['dxcx_condiP'].values[0][1].shape, pid_dx_cx_train['dxcx_condiP'].values[0][1])

    return pid_dx_cx_train


def  get_dxcx_prior_guide():
    for has_CIs in [False]:
        ci_flag = 'wtCI' if has_CIs else 'woCI'
        # ls_tmp=[]
        for seed in range(10):
            for cv_i in range(10):
                pID_cxBucket_cxIdx = pickle.load(open( dir+'/GCT2023_data/data_xgb_GCT/pID_cxBucket_cxIdx___%s_seed%d_cv%d.pkl' % (ci_flag, seed, cv_i), 'rb'))
                dic_cxBucket2Idx = pickle.load(open(dir+'/GCT2023_data/data_xgb_GCT/dic_cxBucket2Idx___%s_seed%d_cv%d.pkl' % (ci_flag, seed, cv_i), 'rb'))
                max_cxBucket_num = pickle.load(open(dir+'/GCT2023_data/data_xgb_GCT/max_cxBucket_num___%s_seed%d_cv%d.pkl' % (ci_flag, seed, cv_i), 'rb'))

                dic_train = pickle.load(open(dir+'/GCT2023_data/data/dic_train_seed%d_%s.pkl' % (seed, ci_flag), 'rb'))[cv_i]
                dic_test = pickle.load(open(dir+'/GCT2023_data/data/dic_test_seed%d_%s.pkl' % (seed, ci_flag), 'rb'))[cv_i]
                #   dic_train       :  [train_ids: np.array, basic_feats: ndarray, train_y: np.array, df_disease_p]
                #   df_disease_p:   ['patient_id', 'ls_dx_ints', 'ls_dx_names', 'dx_prior_indices', 'dx_prior_values']
                dis2no = pickle.load(open(dir+'/GCT2023_data/data/dis2no.pkl', "rb"))
                len_diseases = len(dis2no)

                print(list(pID_cxBucket_cxIdx.columns))
                print(pID_cxBucket_cxIdx)

                print(list(dic_train[-1].columns))
                print(dic_train[-1])


                pid_dx_cx_train=pd.merge(dic_train[-1][['patient_id', 'ls_dx_ints']], pID_cxBucket_cxIdx[['patient_id', 'cx_bucket_idx']], on='patient_id')
                pid_dx_cx_all = pd.merge(pd.concat([dic_train[-1][['patient_id', 'ls_dx_ints']],dic_test[-1][['patient_id', 'ls_dx_ints']]], axis=0).reset_index(drop=True),
                                           pID_cxBucket_cxIdx[['patient_id', 'cx_bucket_idx']], on='patient_id')

                dx_cx_mtx, cx_dx_mtx = get_dx_cx_condiP(pid_dx_cx_train, len(dic_cxBucket2Idx), len_diseases)

                # 构建每个患者的dx_cx_condiP矩阵和cx_dx_condiP矩阵
                max_code_dx=30
                max_code_cx=max_cxBucket_num
                pid_dx_cx_all = get_p_dxcx_condiP(pid_dx_cx_all, dx_cx_mtx, cx_dx_mtx, max_code_dx, max_code_cx)

                # print(list(pid_dx_cx_train.columns),pid_dx_cx_train)
                pid_dx_cx_all=pid_dx_cx_all[['patient_id', 'dxcx_condiP']]
                print(pid_dx_cx_all)
                pickle.dump(pid_dx_cx_all, open(
                    dir+'/GCT2023_data/data_xgb_GCT/pid_dx_cx_all___%s_seed%d_cv%d.pkl' % (ci_flag, seed, cv_i),
                    'wb'))

# dir='E:/0_YangPing/GCT_2023'
dir='F:'
if __name__ == "__main__":
    get_feat_split()    # 得到每个特征的分裂点表
    get_bucket()        # 得到每个人的分桶后特征

    map_bucket_to_ids()   # 统计每个人的分桶特征的最大数量，map到数字

    get_imputed_ids_and_masks()  # 将cx_bucket_idx补齐，并生成mask矩阵（插补的位置为0）
     #    # ['patient_id', 'cx_bucket', 'cx_bucket_idx', 'ls_cx_ints_imputed', 'cx_mask']

    get_cx_prior_guide()   # 得到cx的 'cx_prior_indices', 'cx_prior_values'
    # pID_cxBucket_cxIdx.columns ['patient_id', 'cx_bucket', 'cx_bucket_idx', 'ls_cx_ints_imputed', 'cx_mask', 'cx_prior_indices', 'cx_prior_values']

    # get_feat_outcome_asso()   # 得到cx-death条件概率矩阵
    # ['patient_id', 'cx_bucket', 'cx_bucket_idx', 'ls_cx_ints_imputed', 'cx_mask', 'cx_prior_indices', 'cx_prior_values',
    # cx_oc, dx_oc, cx_oc_imputed, dx_oc_imputed]


    # get_dxcx_prior_guide()   # 得到dx_cx的条件概率矩阵   ['patient_id', 'dxcx_condiP']    患者的(dx-cx条件概率矩阵，cx-dx条件概率矩阵)，已补齐