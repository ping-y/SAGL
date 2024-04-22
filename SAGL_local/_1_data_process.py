import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
# from missingpy import MissForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pickle


def basic_feat_process(path, del_flag, save_ids, seed, has_CIs):
    # path = 'F:/0-数据-CRC/20221229-数据处理/data-2378人/dacca_features_2378p.xlsx'
    df = pd.read_excel(path)

    if del_flag:   #去除没有历史共病记录的患者
        df=pd.merge(save_ids, df, on=['patient_id'])

    print('患者人数：', df['patient_id'].drop_duplicates().shape[0])  # 患者人数： 2360
    print(df.info())   # 包含共病记录的患者人数：1962； 五年死亡率： 0.117

    # 处理标签
    df['OS'] = (pd.to_datetime(df['随访终期'], format='%Y-%m-%d') - pd.to_datetime(df['手术日期'], format='%Y-%m-%d')).dt.days
    # print(df['OS'])
    df['label'] = df['生存状态'].apply(lambda x: 1 if x == '癌性死亡' else 0)

    df['label_5year'] = df['OS'].apply(lambda x: 1 if x <= 365 * 5 else 0)
    df['label_5year'] = df['label_5year'] + df['label']
    df['label_5year'] = df['label_5year'].apply(lambda x: 1 if x == 2 else 0)
    print('总死亡率：', sum(df['label']) / df.shape[0])  # 0.200
    print('五年死亡率：', sum(df['label_5year']) / df.shape[0])  # 0.114

    # 处理特征
    # df['肥胖']=df['肥胖'].apply(lambda x: 1 if pd.notna(x) and x!=0 else x)
    df['婚姻'] = df['婚姻'].apply(lambda x: 1 if pd.notna(x) and x != '0' else (0 if pd.notna(x) else np.NaN))
    df['糖尿'] = df['糖尿'].apply(lambda x: 1 if pd.notna(x) and x != '0' else (0 if pd.notna(x) else np.NaN))
    df['高血'] = df['高血'].apply(lambda x: 1 if pd.notna(x) and x != '0' else (0 if pd.notna(x) else np.NaN))
    df['心功'] = df['心功'].apply(lambda x: np.NaN if pd.notna(x) and x == 'xg' else x)
    df['心功'] = df['心功'].apply(lambda x: float(x))
    # df['贫血'] = df['贫血'].apply(lambda x: int(x))
    df['肿瘤部位'] = df['肿瘤部位'].apply(lambda x: 0 if x >= 11 else 1)  # 0：结肠 （11-21）；1：直肠 （1-10）

    dic_xingtai = {'溃疡': 0, '肿块型': 1, '菜花型': 2, '隆起型': 3, '息肉样': 4}
    df['肿瘤形态'] = df['肿瘤形态'].apply(lambda x: dic_xingtai[x] if pd.notna(x) else x)

    df['大小-大'] = df['大小'].apply(lambda x: max([float(x.split('*')[0]), float(x.split('*')[1])]) if pd.notna(x) else x)
    df['大小-小'] = df['大小'].apply(lambda x: min([float(x.split('*')[0]), float(x.split('*')[1])]) if pd.notna(x) else x)

    dic_fenhua = {'低': 0, '中': 1, '高': 2}
    df['分化'] = df['分化'].apply(lambda x: dic_fenhua[x] if pd.notna(x) and x in dic_fenhua else np.NaN)

    df['阳性淋巴数量'] = df['淋巴比率'].apply(lambda x: float(x.split('/')[0]) if pd.notna(x) else x)
    df['T'] = df['T病'].apply(
        lambda x: 'T2' if pd.notna(x) and x == 'T2x' else ('T0' if pd.notna(x) and x == 'Tx' else x))
    df['N'] = df['N病'].apply(
        lambda x: 'N2' if pd.notna(x) and x == 'N2x' else ('N0' if pd.notna(x) and x == 'Nx' else x))
    df['M'] = df['M病'].apply(
        lambda x: 'M1c' if pd.notna(x) and x == 'M1cx' else ('M0' if pd.notna(x) and x == 'Mx' else x))
    for i in ['T', 'N', 'M']:
        uniq_elem = list(df[i].unique()[np.where(pd.notna(df[i].unique()))])
        dic_tmp = dict(zip(uniq_elem, [j for j in range(len(uniq_elem))]))
        df[i] = df[i].apply(lambda x: dic_tmp[x] if pd.notna(x) else x)

    dic_ssxz = {'Rx': 0, 'R0': 1, 'R1': 2, 'R2~R5': 3}
    df['手术性质'] = df['手术性质'].apply(lambda x: dic_ssxz[x] if pd.notna(x) and x in dic_ssxz else np.NaN)

    # 添加CI特征
    if has_CIs:
        pat_CIs = pd.read_csv('F:/0-数据-CRC/R_CI/patient_CI.csv')
        # print('pat_CIs', pat_CIs)
        df = pd.merge(df, pat_CIs, on=['patient_id'])
        # print('df---', df)
        continuous_feats = ['年龄', 'BMI', '肥胖', '心功', '贫血', '腹水', '评分', '缘距', '大小-大', '大小-小', '分化', '阳性淋巴数量', '术时', '初期CEA', 'cci', 'eci', 'c3']
    else:
        continuous_feats = ['年龄', 'BMI', '肥胖', '心功', '贫血', '腹水', '评分', '缘距', '大小-大', '大小-小', '分化', '阳性淋巴数量', '术时', '初期CEA']    # , 'cci', 'eci', 'c3'
    discrete_feats = ['性别', '婚姻', '肿瘤部位', '肿瘤形态', '肿瘤性质', '梗阻', '套叠', '穿孔', 'T', 'N', 'M', '手术性质', '造口', '扩大', '新辅类型',
                      '口靶']  # '糖尿','高血'
    total_feats = continuous_feats.copy()
    total_feats.extend(discrete_feats)

    cols = ['patient_id']
    cols.extend(continuous_feats)
    cols.extend(discrete_feats)
    cols.extend(['label', 'label_5year', 'OS'])
    df = df[cols]
    print()
    print(df.info())

    # 去掉缺失率大于20%的特征
    del_feats = []
    for i in df.columns:
        if pd.isna(df[i]).sum() > 0.2 * df.shape[0]:
            del_feats.append(i)

            if i in continuous_feats:
                continuous_feats.remove(i)
            elif i in discrete_feats:
                discrete_feats.remove(i)
            if i in total_feats:
                total_feats.remove(i)

    print(del_feats)
    df = df[[i for i in df.columns if i not in del_feats]]

    # 划分数据集-交叉验证-10折-10次
    skf = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
    y = df['label_5year']
    dic_train= dict()
    dic_test = dict()

    for i, (train_index, test_index) in enumerate(skf.split(np.zeros(y.shape[0]), y)):
        # print('train_index',train_index)
        # print('test_index', test_index)

        train_ids=df['patient_id'].loc[train_index].values
        test_ids=df['patient_id'].loc[test_index].values

        dic_feats2colNum = dict(zip(total_feats, [j for j in range(len(total_feats))]))
        train_nda = df[total_feats].loc[train_index].values  # 不含标签，只包含特征
        test_nda = df[total_feats].loc[test_index].values  # 不含标签，只包含特征

        train_y = df['label_5year'].loc[train_index].values
        test_y = df['label_5year'].loc[test_index].values

        # print(total_feats)
        # 填补缺失值

        # print('missforest training ......')
        # imputer = MissForest(max_iter=1, class_weight='balanced',random_state=1337)   # n_jobs=3,
        # imputer.fit(train_nda, cat_vars=[dic_feats2colNum[c] for c in discrete_feats])
        # train_nda=imputer.transform(train_nda)
        # test_nda = imputer.transform(test_nda)

        # 插补缺失值
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        train_nda[:, [dic_feats2colNum[c] for c in continuous_feats]] = \
            imp_mean.fit_transform(train_nda[:, [dic_feats2colNum[c] for c in continuous_feats]])
        test_nda[:, [dic_feats2colNum[c] for c in continuous_feats]] = \
            imp_mean.fit_transform(test_nda[:, [dic_feats2colNum[c] for c in continuous_feats]])

        imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        train_nda[:, [dic_feats2colNum[c] for c in discrete_feats]] = imp_mean.fit_transform(
            train_nda[:, [dic_feats2colNum[c] for c in discrete_feats]])
        test_nda[:, [dic_feats2colNum[c] for c in discrete_feats]] = imp_mean.fit_transform(
            test_nda[:, [dic_feats2colNum[c] for c in discrete_feats]])

        # 类别变量onehot
        # 剔除占比特别小的类别（T、N、M、造口==5）
        train_df = pd.DataFrame(train_nda, columns=total_feats)
        test_df = pd.DataFrame(test_nda, columns=total_feats)
        X_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
        discrete_feats_onehot = []
        for cc in discrete_feats:
            enc = OneHotEncoder()
            enc.fit(X_df[cc].values.reshape(-1, 1))

            tmp_df = enc.transform(X_df[cc].values.reshape(-1, 1)).toarray()

            new_cols = [cc + str(j) for j in range(tmp_df.shape[-1])]
            discrete_feats_onehot.extend(new_cols)

            tmp_df = pd.DataFrame(tmp_df, columns=new_cols)
            del X_df[cc]
            X_df = pd.concat([X_df, tmp_df], axis=1)

        # 删去人数特别少的类别
        for cc in discrete_feats_onehot.copy():
            if sum(X_df[cc]) < 10:
                del X_df[cc]
                discrete_feats_onehot.remove(cc)
        train_df = X_df.iloc[:train_df.shape[0]]
        test_df = X_df.iloc[train_df.shape[0]:]

        dic_train[i]=[train_ids, train_df, train_y]
        dic_test[i]=[test_ids, test_df, test_y]

    return dic_train, dic_test

def agg_disease(df_group):
    d_set=set()
    for i in df_group['ALL_DISEASE_2']:
        d_set|=i
    return ", ".join(list(d_set))


def get_edge_num(x,sign_RRs):
    x=list(x)
    count=0
    for i in range(len(x)-1):
        for j in range(i+1,len(x)):
            if [x[i],x[j]] in sign_RRs or [x[j],x[i]] in sign_RRs:
                count+=1
    return count

#
# def get_edge_indice_value(x,dis2no,dic_edge2RR):
#     indices=[]
#     values=[]
#     x = list(x)
#     for i in range(len(x)-1):
#         for j in range(i+1,len(x)):
#             if (x[i],x[j]) in dic_edge2RR or (x[j],x[i]) in dic_edge2RR:
#                 indices.extend([(dis2no[x[i]], dis2no[x[j]]), (dis2no[x[j]], dis2no[x[i]]) ])
#                 values.extend([dic_edge2RR[(x[i],x[j])],dic_edge2RR[(x[i],x[j])]])
#     return [indices,values]


def get_edge_indice_value(x, dic_edge2RR):
    indices=[]
    values=[]
    x = list(x)
    for i in range(len(x)-1):
        for j in range(i+1,len(x)):
            if (x[i],x[j]) in dic_edge2RR or (x[j],x[i]) in dic_edge2RR:
                indices.extend([ (i,j), (j,i) ])
                values.extend([dic_edge2RR[(x[i],x[j])],dic_edge2RR[(x[i],x[j])]])
    return [indices,values]



if __name__ == "__main__":
    # for edge in [('RR','0.010000'),('RR','0.001000'),('CC','0.010000'),('CC', '0.001000'),('phi','0.010000'),('phi','0.001000')]:
    for edge in [('phi','0.000500_caseDis0_chronic0')]:    # ('cc0.001_and_phi0.001','_binary'), ('cc0.001_and_phi0.001_and_rr0.001largerer2','_binary')
        for has_CIs in [False]:
            ci_flag = 'wtCI' if has_CIs else  'woCI'

            for seed in range(10):

                process=[1,2,3,4]
                # process = [5]
                if 1 in process:
                    path = 'F:/0-数据-CRC/20221229-数据处理/data-2378人/dacca_features_2378p.xlsx'

                    del_flag=True   # 是否去掉没有历史共病记录的病人
                    save_ids = pd.read_excel('F:/0-数据-CRC/20221229-数据处理/data-2378人/术前疾病_2378p.xlsx')
                    save_ids = save_ids[['patient_id']].drop_duplicates().reset_index(drop=True)

                    dic_train, dic_test=basic_feat_process(path, del_flag, save_ids, seed, has_CIs)
                    # dic_test:
                    # {0:   patient_id: [array([1550, 2053, 2735,.....
                    #         feat_df: [236 rows x 75 columns]    -   columns : ['年龄', 'BMI', '肥胖', '贫血', '腹水', ... ,  '性别0', '性别1', '婚姻0', ...]
                    #         label:  array([0, 0, 0, 0, 1,
                    print('dic_train.keys',dic_train.keys())


                    # pickle.dump(dic_train, open('F:/GCT2023_data/data/dic_train_forgeNet_seed%d_%s.pkl'%(seed,ci_flag), 'wb'), protocol=4)
                    # pickle.dump(dic_test, open('F:/GCT2023_data/data/dic_test_forgeNet_seed%d_%s.pkl'%(seed,ci_flag), 'wb'), protocol=4)


                if 2 in process:
                    # 先统计一下，每个患者有多少个疾病、边在疾病网络中
                    df_disease=pd.read_excel('F:/0-数据-CRC/20221229-数据处理/data-2378人/术前疾病_2378p.xlsx')
                    df_disease=df_disease[['patient_id', 'ALL_DISEASE_2']]
                    df_disease['ALL_DISEASE_2']=df_disease['ALL_DISEASE_2'].apply(lambda x: x.split(', '))
                    df_disease['ALL_DISEASE_2']=df_disease['ALL_DISEASE_2'].apply(lambda x: set([i[:3] for i in x]))

                    print(df_disease)

                    df_disease=pd.DataFrame([[i[0],i[1]] for i in df_disease.groupby(['patient_id'],as_index=False).apply(agg_disease).values],columns=['patient_id', 'ALL_DISEASE_2'])
                    df_disease['ALL_DISEASE_2'] = df_disease['ALL_DISEASE_2'].apply(lambda x: set(x.split(', ')))
                    print('有共病的人数：',df_disease.shape[0])   # 1962
                    print(df_disease)

                    # 将不包含共病信息的患者加入df_disease
                    pid_wo_comor=list(set(list(np.concatenate([dic_train[0][0],dic_test[0][0]])))-set(list(df_disease['patient_id'].values)))
                    pid_wo_comor=pd.DataFrame(zip(pid_wo_comor,[set() for _ in range(len(pid_wo_comor))]),columns=['patient_id', 'ALL_DISEASE_2'])
                    df_disease=pd.concat([df_disease, pid_wo_comor], axis=0).reset_index(drop=True)

                    # 删去患病率低于一定阈值的疾病
                    diss = pd.read_excel('F:/0-数据-CRC/20221229-数据处理/data-2378人/统计手术及以前的疾病患病率.xlsx')
                    diss['ICD3'] = diss['疾病'].apply(lambda x: x[:3])
                    diss = diss[['ICD3', '患病率']].groupby('ICD3')['患病率'].apply(sum)
                    save_dis=set(list(diss[diss >= 0.001].index))

                    df_disease['save_dis'] = df_disease['ALL_DISEASE_2'].apply(lambda x: x & save_dis)
                    df_disease['save_dis_num'] = df_disease['save_dis'].apply(lambda x: len(x))
                    print("df_disease['save_dis_num'].describe()",df_disease['save_dis_num'].describe())

                    # edges=pd.read_csv('F:/0-数据-CRC/20221229-数据处理/data-2378人/CRC网络-total_edges_.csv')
                    # edges = pd.read_csv('F:/0-数据-CRC/20221229-数据处理/data-2378人/CRC网络-edge_list_0.010000.csv')
                    if edge[0]+edge[1] in ['cc0.001_and_phi0.001_binary' ,'cc0.001_and_phi0.001_and_rr0.001largerer2_binary',
                           'cc0.001_and_phi0.001_and_rr0.01_binary', 'cc0.001_and_phi0.0005_binary' ]:
                        edges = pd.read_csv('F:/0-数据-CRC/20221229-数据处理/data-2378人/%s%s.csv' % (edge[0], edge[1]))
                    else:
                        edges = pd.read_csv('F:/0-数据-CRC/20221229-数据处理/data-2378人/%s_edge_list_%s.csv'%(edge[0],edge[1]))


                    node_set = set(list(edges['disA'].values)) | set(list(edges['disB'].values))
                    # sign_RRs=list([list(i) for i in edges[edges['RR_flg']=='comorbid'][['disA','disB']].values])
                    sign_RRs = list([list(i) for i in edges[['disA', 'disB']].values])

                    df_disease['ntw_disease'] = df_disease['save_dis'].apply(lambda x: x&node_set)
                    df_disease['ntw_disease_num']= df_disease['ntw_disease'].apply(lambda x: len(x))
                    print("df_disease['ntw_disease_num'].describe()",df_disease['ntw_disease_num'].describe())
                    # mean        6.783384           # std         4.526028
                    # min  1        # 25%   3        # 50%   6       # 75%  9        # max  30


                    for i in range(14):
                        print('个数为',i,'的人数：',df_disease[df_disease['ntw_disease_num']==i].shape[0])
                        # 个数为 0 的人数： 842           # 个数为 1 的人数： 520            # 个数为 2 的人数： 287            # 个数为 3 的人数： 152
                        # 个数为 4 的人数： 80            # 个数为 5 的人数： 44            # 个数为 6 的人数： 15
                        # 个数为 7 的人数： 10            # 个数为 8 的人数： 4            # 个数为 9 的人数： 2
                        # 个数为 10 的人数： 2            # 个数为 11 的人数： 1            # 个数为 12 的人数： 1            # 个数为 13 的人数： 2

                        # 0.01:
                        # 个数为 0 的人数： 326        # 个数为 1 的人数： 199        # 个数为 2 的人数： 233        # 个数为 3 的人数： 267
                        # 个数为 4 的人数： 227        # 个数为 5 的人数： 222        # 个数为 6 的人数： 168        # 个数为 7 的人数： 120
                        # 个数为 8 的人数： 72        # 个数为 9 的人数： 48        # 个数为 10 的人数： 29

                    df_disease['edge_num'] = df_disease['ntw_disease'].apply(lambda x: get_edge_num(x,sign_RRs))
                    print(df_disease['edge_num'].describe())
                    for i in range(11):
                        print('个数为',i,'的人数：',df_disease[df_disease['edge_num']==i].shape[0])
                        # 个数为 0 的人数： 1394            # 个数为 1 的人数： 259            # 个数为 2 的人数： 34            # 个数为 3 的人数： 114
                        # 个数为 4 的人数： 7            # 个数为 5 的人数： 16            # 个数为 6 的人数： 59            # 个数为 7 的人数： 2
                        # 个数为 8 的人数： 1            # 个数为 9 的人数： 10            # 个数为 10 的人数： 29

                        # 0.01
                        # 个数为 0 的人数： 545        # 个数为 1 的人数： 223        # 个数为 2 的人数： 54        # 个数为 3 的人数： 213
                        # 个数为 4 的人数： 23        # 个数为 5 的人数： 60        # 个数为 6 的人数： 150        # 个数为 7 的人数： 15
                        # 个数为 8 的人数： 40        # 个数为 9 的人数： 58        # 个数为 10 的人数： 100

                    # 提取疾病特征
                    # [patient_id, ls_dx_ints, ls_dx_names, dx_prior_indices, dx_prior_values]
                    #  ls_dx_ints: [0,5,8]  ls_dx_names: ['I20',...]
                    #  dx_prior_indices: [[0,5], [0,51], [5,0], [51,0]]
                    #  dx_prior_values: [0.153, 0.146, ...... ]
                    # dis2no=dict(zip(sorted(node_set), [i for i in range(len(node_set))]))
                    # RR_w=[list(i) for i in edges[edges['RR_flg'] == 'comorbid'][['disA', 'disB', 'RR']].values]
                    RR_w = [list(i) for i in edges[['disA', 'disB', 'RR']].values]
                    weight=[i[-1] for i in RR_w]
                    max_RR=max(weight)
                    min_RR = min(weight)
                    dic_edge2RR=dict()
                    for i in RR_w:
                        if edge[0]+edge[1] in ['cc0.001_and_phi0.001_binary' ,'cc0.001_and_phi0.001_and_rr0.001largerer2_binary',
                                'cc0.001_and_phi0.001_and_rr0.01_binary','cc0.001_and_phi0.0005_binary' ]:
                            dic_edge2RR[(i[0], i[1])] =i[2]
                            dic_edge2RR[(i[1], i[0])] =i[2]
                        else:
                            dic_edge2RR[(i[0],i[1])]=(i[2]-min_RR)/(max_RR-min_RR)
                            dic_edge2RR[(i[1],i[0])]=(i[2]-min_RR)/(max_RR-min_RR)

                    # df_disease['ls_dx_names']=df_disease['ntw_disease'].apply(lambda x: list(x))
                    # df_disease['ls_dx_ints']=df_disease['ls_dx_names'].apply(lambda x: [dis2no[i] for i in x])
                    # df_disease['prior']=df_disease['ntw_disease'].apply(lambda x: get_edge_indice_value(x,dis2no,dic_edge2RR))
                    # df_disease['dx_prior_indices']=df_disease['prior'].apply(lambda x: x[0])
                    # df_disease['dx_prior_values']=df_disease['prior'].apply(lambda x: x[1])

                    dis2no = dict(zip(sorted(list(save_dis)), [i for i in range(len(save_dis))]))

                    df_disease['ls_dx_names']=df_disease['save_dis'].apply(lambda x: sorted(list(x)))   # 保留患病率大于0.001的疾病（至少有两个人患）
                    df_disease['ls_dx_ints'] = df_disease['ls_dx_names'].apply(lambda x: [dis2no[i] for i in x])
                    df_disease['prior'] = df_disease['ls_dx_names'].apply(lambda x: get_edge_indice_value(x, dic_edge2RR))
                    df_disease['dx_prior_indices']=df_disease['prior'].apply(lambda x: x[0])
                    df_disease['dx_prior_values']=df_disease['prior'].apply(lambda x: x[1])

                    print(df_disease[df_disease['edge_num']>0]['dx_prior_values'])


                    for p in list(dic_train.keys()):  # [train_ids: np.array, train_df: df, train_y: np.array]
                        df_disease_p=pd.merge(pd.DataFrame(dic_train[p][0],columns=['patient_id']),
                                              df_disease[['patient_id', 'ls_dx_ints', 'ls_dx_names', 'dx_prior_indices', 'dx_prior_values']],on='patient_id').reset_index(drop=True)
                        dic_train[p].append(df_disease_p)

                    for p in list(dic_test.keys()):
                        df_disease_p=pd.merge(pd.DataFrame(dic_test[p][0],columns=['patient_id']),
                                              df_disease[['patient_id', 'ls_dx_ints', 'ls_dx_names', 'dx_prior_indices', 'dx_prior_values']],on='patient_id').reset_index(drop=True)
                        dic_test[p].append(df_disease_p)

                if 3 in process:
                    # 标准化基本特征
                    for p in list(dic_train.keys()):  # [train_ids: np.array, train_df: df, train_y: np.array, df_disease_p]
                        df_basic_feats=dic_train[p][1]
                        df_basic_feats_test=dic_test[p][1]

                        scaler = StandardScaler()
                        scaler.fit(df_basic_feats)
                        df_basic_feats=scaler.transform(df_basic_feats)
                        df_basic_feats_test=scaler.transform(df_basic_feats_test)

                        dic_train[p][1]=df_basic_feats
                        dic_test[p][1]=df_basic_feats_test

                if 4 in process:
                    # 存一下预处理的特征dic_train, dic_test
                    pickle.dump(dis2no, open('F:/GCT2023_data/data/dis2no_seed%d_%s_DCN_%s_%s.pkl'%(seed,ci_flag,edge[0],edge[1]), 'wb'), protocol=4)
                    pickle.dump(dic_train, open('F:/GCT2023_data/data/dic_train_seed%d_%s_DCN_%s_%s.pkl'%(seed,ci_flag,edge[0],edge[1]), 'wb'), protocol=4)
                    pickle.dump(dic_test, open('F:/GCT2023_data/data/dic_test_seed%d_%s_DCN_%s_%s.pkl'%(seed,ci_flag,edge[0],edge[1]), 'wb'), protocol=4)
                    # #   dic_train       :  [train_ids: np.array, basic_feats: ndarray, train_y: np.array, df_disease_p]
                    # #   df_disease_p:   ['patient_id', 'ls_dx_ints', 'ls_dx_names', 'dx_prior_indices', 'dx_prior_values']

                if 5 in process:   #进一步划分出验证集
                    trainIDs_ls = []
                    valIDs_ls = []
                    testIDs_ls = []

                    dic_train = pickle.load(open('F:/GCT2023_data/data/dic_train_seed%d_%s_DCN_%s_%s.pkl'%(seed,ci_flag,edge[0],edge[1]), "rb"))
                    dic_test = pickle.load(open('F:/GCT2023_data/data/dic_test_seed%d_%s_DCN_%s_%s.pkl'%(seed,ci_flag,edge[0],edge[1]), "rb"))
                    for cv_fold in range(10):  # 十折交叉验证
                        train_ls = dic_train[cv_fold]
                        test_ls = dic_test[cv_fold]

                        # 从train中划分出验证集val
                        num = test_ls[0].shape[0]
                        tmp = pd.DataFrame(zip(train_ls[0], train_ls[2]), columns=['patient_id', 'y'])
                        posIDs = tmp[tmp['y'] == 1]['patient_id'].values
                        negIDs = tmp[tmp['y'] == 0]['patient_id'].values
                        np.random.seed(0)
                        posIDs = np.random.choice(posIDs, int(posIDs.shape[0] / tmp.shape[0] * num), replace=False)
                        np.random.seed(0)
                        negIDs = np.random.choice(negIDs, int(negIDs.shape[0] / tmp.shape[0] * num), replace=False)
                        valIDs = list(posIDs)
                        valIDs.extend(list(negIDs))
                        trainIDs= [i for i in train_ls[0] if i not in valIDs]
                        trainIDs_ls.append(trainIDs)
                        valIDs_ls.append( [i for i in train_ls[0] if i not in trainIDs])
                        testIDs_ls.append(list(test_ls[0]))

                        pickle.dump(trainIDs_ls, open('F:/GCT2023_data/data/trainIDs_ls_seed%d_%s.pkl'%(seed,ci_flag), 'wb'), protocol=4)
                        pickle.dump(valIDs_ls, open('F:/GCT2023_data/data/valIDs_ls_seed%d_%s.pkl'%(seed,ci_flag), 'wb'), protocol=4)
                        pickle.dump(testIDs_ls, open('F:/GCT2023_data/data/testIDs_ls_seed%d_%s.pkl'%(seed,ci_flag), 'wb'), protocol=4)

                if 6 in process:    # 按年份生成标签

                    del_flag = True  # 是否去掉没有历史共病记录的病人
                    save_ids = pd.read_excel(dir+'/20221229-数据处理/data-2378人/术前疾病_2378p.xlsx')
                    save_ids = save_ids[['patient_id']].drop_duplicates().reset_index(drop=True)
                    df = pd.read_excel(dir+'/20221229-数据处理/data-2378人/dacca_features_2378p.xlsx')

                    if del_flag:  # 去除没有历史共病记录的患者
                        df = pd.merge(save_ids, df, on=['patient_id'])

                    print('患者人数：', df['patient_id'].drop_duplicates().shape[0])  # 患者人数： 2360
                    # print(df.info())  # 包含共病记录的患者人数：1962； 五年死亡率： 0.117

                    # 处理标签
                    df['OS'] = (pd.to_datetime(df['随访终期'], format='%Y-%m-%d') - pd.to_datetime(df['手术日期'],
                                                                                                   format='%Y-%m-%d')).dt.days
                    # print(df['OS'])
                    df['label'] = df['生存状态'].apply(lambda x: 1 if x == '癌性死亡' else 0)

                    df['label_5year'] = df['OS'].apply(lambda x: 1 if x <= 365 * 5 else 0)
                    df['label_5year'] = df['label_5year'] + df['label']
                    df['label_5year'] = df['label_5year'].apply(lambda x: 1 if x == 2 else 0)
                    print('总死亡率：', sum(df['label']) / df.shape[0])  # 0.200
                    print('五年死亡率：', sum(df['label_5year']) / df.shape[0])  # 0.114


                    # 10年死亡人数，死亡率
                    col_nm = 'label_10year'
                    year = 10
                    df[col_nm] = df['OS'].apply(lambda x: 1 if x <= 365 * year else 0)
                    df[col_nm] = df[col_nm] + df['label']
                    df[col_nm] = df[col_nm].apply(lambda x: 1 if x == 2 else 0)
                    print('10年死亡人数， 10年死亡率：', sum(df[col_nm]), sum(df[col_nm]) / df.shape[0])

                    # 3年死亡人数，死亡率
                    col_nm = 'label_3year'
                    year = 3
                    df[col_nm] = df['OS'].apply(lambda x: 1 if x <= 365 * year else 0)
                    df[col_nm] = df[col_nm] + df['label']
                    df[col_nm] = df[col_nm].apply(lambda x: 1 if x == 2 else 0)
                    print('3年死亡人数， 3年死亡率：', sum(df[col_nm]), sum(df[col_nm]) / df.shape[0])

                    print(df[['patient_id','label','label_5year','label_10year','label_3year']])
                    pickle.dump(df[['patient_id','label','label_5year','label_10year','label_3year']],
                                open(dir+'/GCT2023_data/data/pID_labels.pkl','wb'))