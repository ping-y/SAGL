import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from scipy.sparse import csr_matrix,coo_matrix
import pickle
import urllib.request
import pandas as pd
import numpy as np

from scipy import sparse
import networkx as nx
from igraph import *
from tqdm import tqdm
import time
import config


def get_onehot_disea(df_p_d):

    def get_onehot_d(df_group,dict_d):
        tmp = np.zeros(len(dict_d))
        for i in df_group[config.ALL_DISEASE]:
            for j in i:
                tmp[dict_d[j]] = 1
        return df_group[config.SFZH].values[0], tmp

    dict_d=set()
    for i in df_p_d[config.ALL_DISEASE]:
        dict_d=dict_d|i
    dict_d = np.sort(np.array(list(dict_d)))
    dict_d=dict(zip(dict_d,range(dict_d.shape[0])))  # 所有疾病-编号的字典

    df_p_d=df_p_d.groupby([config.SFZH]).apply(get_onehot_d, dict_d)
    df_p_d=dict(df_p_d.values)
    df_p_d = pd.DataFrame(zip(df_p_d.keys(), df_p_d.values()), columns=[config.SFZH, 'lst_disea_binary'])

    return df_p_d, dict_d



def get_sex_chronic_disease(chronic,sex):
    #对疾病进行筛选，得到符合条件的疾病 慢性和性别特异性疾病 sex=1 代表要考虑性别特异性疾病
    """两个字典 chronic=1 返回慢病字典 sex=1 返回性别特异性字典  1代表男性 2代表女性 0代表不是性别特异性"""
    df=pd.read_csv("dis_sex_chronic.csv",encoding="gbk")
        #大于0 是宽松的条件， 等于1 是严格的条件，我们暂定宽松的条件
    if (chronic==1  and sex==0):#性别的1是男性 2代码女性特异性
        df_ans=df[ df.chronic>0]
        set_chronic=set(list(df_ans["dis"].values))
        return set_chronic  # 返回慢病集合

    else:
        df_ans=df[ df.SexDisease==1]#男性特异性
        dic={}
        for i in df_ans["dis"].values:
            dic[i]=1
        df_ans=df[ df.SexDisease==2]#女性特异性
        for i in df_ans["dis"].values:
            dic[i]=2
        return dic  # 能够查询代表满足要求


def construct_sparse_matrix(p_d_nda, dict_all_d,is_chronic):
    """
    输入参数df[['SFZH','diseases']]
    该函数输出的结果：字典：dic_cols,dic_rows；
    存储pkl文件 稀疏矩阵：patient_disease_csr_matrix
    功能：根据从数据库读入的数据生成患者-慢病稀疏矩阵，（一人多条记录合并为一条记录）
    """
    # 是否去除急性病  （1表示去除急性病）
    if is_chronic==1:
        selected_d=set(list(dict_all_d.keys()))
        set_chornic = get_sex_chronic_disease(1, 0)  # 得到慢病字典； keys是慢性疾病
        selected_d.intersection_update(set_chornic)  # 移除disease中不属于dic的元素 ，得到的是慢病集合
        print('selected_d',len(selected_d))   #乱序的
        # print('-----dict_all_d-----', dict_all_d)   # 有序的
        selected_d=np.sort(np.array(list(selected_d)))  # 对selected_d排序
        tmp_dic=dict()
        for i in selected_d:
            tmp_dic[i]=dict_all_d[i]
        p_d_nda=p_d_nda[:, list(tmp_dic.values())]   # 选取慢病列
        dict_all_d=dict(zip(list(tmp_dic.keys()),range(len(tmp_dic))))  #更新列映射字典
        # print('-----dict_all_d-----',len(dict_all_d),dict_all_d)
    # 构建稀疏矩阵
    p_d_nda = sparse.csr_matrix(p_d_nda)
    print('construct_sparse_matrix——dict_all_d.shape,p_d_nda.shape', len(dict_all_d), p_d_nda.shape)
    return p_d_nda, dict_all_d



def choose_diseases_based_on_matrix(p_d_nda, dict_selected_d, has_IHD, prevalence_threshold, lst_research_d):
    """
    处理稀疏矩阵的列，去除流行率小于1%的疾病
    输入参数：
        _num_male,nem_female为纳入患者中的男女人数，由函数construct_sex_count_dict()可计算得到
        _flag： flag为1，则去除I20-I25，共病网络不考虑这六种疾病 ; flag=0,则将I20-I25纳入共病网络
        _prevalence_threshold:选取的疾病的流行率下限
        _save_dir:读取及保存文件的文件夹名字
    输出文件：csc_matrix_final.pkl 去除了流行率小于1%的列后的新矩阵；dic_cols_new.pkl 新的疾病-列映射
    返回值：_dic_disease_prevalence   返回疾病流行度字典，与稀疏矩阵的列顺序对应
    """
    print("除去流行程度小于%.5f的疾病---"%(prevalence_threshold))
    pastt=time.time()

    if has_IHD==0:
        # 去除I20-I25
        for i in lst_research_d:
            if i in dict_selected_d:
                del dict_selected_d[i]

    col_names_new=[]
    dic_disease_prevalence={}
    dic_disease_prevalence_rate={}
    num_patient=p_d_nda.shape[0]

    for key in dict_selected_d:
        prevalence=p_d_nda[:,dict_selected_d[key]].sum()
        prevalence_rate=p_d_nda[:,dict_selected_d[key]].sum()/num_patient

        if (prevalence_rate>prevalence_threshold):
            col_names_new.append(key)
            dic_disease_prevalence[key]=prevalence
            dic_disease_prevalence_rate[key]=prevalence_rate

    # 生成新的稀疏矩阵，去除了流行率小于1%的疾病后
    col_names_new = dict(zip(col_names_new, range(len(col_names_new))))  # 新的疾病-列映射
    csc_matrix_final=p_d_nda[:,[dict_selected_d[i] for i in list(col_names_new.keys())]]
    print("去除流行率小于%.5f的疾病后，稀疏矩阵的维度："%(prevalence_threshold),csc_matrix_final.shape)

    # print('col_names_new',col_names_new)  #有序的

    preva_df = pd.DataFrame(dic_disease_prevalence_rate, index=[1])
    preva_df.sort_values(by=1, axis=1, ascending=False,inplace=True)
    preva_df=preva_df.iloc[:,0:10]
    print("流行率前10的疾病：")
    for i in preva_df:
        print(i, preva_df.loc[1][i])

    return  csc_matrix_final, col_names_new , dic_disease_prevalence,dic_disease_prevalence_rate, num_patient   # 返回疾病流行度字典，与稀疏矩阵的列顺序对应



def compute_Cij(dic_cols,csc_matrix_final):
    """
    计算Cij矩阵,是一个上三角矩阵，同时患i和j两种疾病的人数，主对称轴元素均为0
    输入参数：dic_cols:疾病-列表映射；
    输入参数：csc_matrix_final:慢病稀疏矩阵
    返回值：Cij矩阵
    """
    print("开始计算Cij--------------------------")
    # print('dic_cols',dic_cols)  有序的
    Cij=np.zeros((len(dic_cols),len(dic_cols)))
    for i in tqdm(range(len(dic_cols))):
        for j in range(i+1,len(dic_cols)):
            two_cols_sum=(csc_matrix_final[:,i]+csc_matrix_final[:,j])
            cij=np.sum(two_cols_sum.data == 2)   #一个人同时患两种病
            Cij[i][j]=cij
    return Cij


def compute_RR_CI(Cij,prevalence,N,dic_cols):
    """
    计算RR值及其置信区间 ,99%的置信区间，置信区间不包含1，则有意义
    输入参数：Cij:上三角矩阵，由函数compute_Cij()计算所得；prevalence:字典，由函数choose_diseases_based_on_matrix()计算可得；N:纳入的总人数；dic_cols：疾病-稀疏矩阵列的映射
    返回值：有意义的边组成的列表edge_list
    返回列表的结构：[[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的RR值，置信下区间，置信上区间,同时患两种疾病的人数],....]
    """
    Cij_num=0
    edge_list=[]
    list_cols_name=list(dic_cols.keys())
    for i in range(Cij.shape[0]):
        for j in range(i+1,Cij.shape[0]):
            if Cij[i][j]!=0:
                Cij_num+=1
                prevalence1 = prevalence[list_cols_name[i]]
                prevalence2 = prevalence[list_cols_name[j]]
                RR_ij=(Cij[i][j]/prevalence1)*(N/prevalence2)
                if RR_ij<0:
                    raise Exception("RR 溢出啦",RR_ij)

                Sigma=1/Cij[i][j]+(1/prevalence1)*(1/prevalence2)-1/N-(1/N)*(1/N)   #会产生除零错误，所以应该在计算前判断Cij是否为零；（Cij为零时，RR值也为零）
                low=RR_ij*np.exp(-1*2.56*Sigma)
                high=RR_ij*np.exp(2.56*Sigma)
                if(RR_ij>1 and low>1):  #这里只考虑了两个节点联系比随机情况下更强的情况
                    edge_list.append([list_cols_name[i],list_cols_name[j],prevalence1,prevalence2,RR_ij,low,high,Cij[i][j]])
                    #上面一行：添加有意义的边到边列表中，[[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的RR值，置信下区间，置信上区间，同时患两种病的人数],....]
    print("不为零的Cij的个数：",Cij_num)
    print("RR，有意义的边数：", len(edge_list))
    return edge_list


def compute_phi_significated(Cij, dic_prevalence, dic_cols_new,N):
    """
    计算phi值及t值 ,99%的置信水平
    输入参数：Cij:上三角矩阵，由函数compute_Cij()计算所得；prevalence:字典，由函数choose_diseases_based_on_matrix()计算可得；N:纳入的总人数；dic_cols：疾病-稀疏矩阵列的映射
    返回值：有意义的边组成的列表edge_list
    返回列表的结构：[[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的phi值，t值，无意义位，同时患两种病的人数],....]
    """
    edge_list=[]
    list_cols_name=list(dic_cols_new.keys())
    for i in range(Cij.shape[0]):
        for j in range(i+1,Cij.shape[0]):
            prevalence1=dic_prevalence[list_cols_name[i]]
            prevalence2 = dic_prevalence[list_cols_name[j]]
            a = (0.1*prevalence1) * (0.1*prevalence2) * (0.1*(N - prevalence1))*(0.1*(N - prevalence2))
            # a = prevalence1 * prevalence2 * (N - prevalence1) * (N - prevalence2)
            if (a) <= 0:
                raise Exception("phi 溢出啦",a)
            phi_ij=((0.1*Cij[i][j])*(0.1*N)-(0.1*prevalence1)*(0.1*prevalence2))/np.sqrt(a)
            t=0  #初始化t
            n=0
            if abs(phi_ij) < 1:  # phi=1时，会发生除零错误,|phi|>1时，会发生计算错误
                n = max(prevalence1, prevalence2)
                # n=N     # 注意测试一下
                t = (phi_ij * math.sqrt(n - 2)) / np.sqrt(1 - (phi_ij ** 2))
            elif phi_ij>1 or phi_ij<-1: # 不会大于1
                print("有phi大于1 或者小于-1 ，考虑截断,phi值为：",phi_ij)
                # 若phi=1，只能是这种情况：A病和B病必定同时出现，且A病和B病不单独出现，这时的phi=1；因为前面步骤去除了流行度小于1%的疾病，所以这种情况基本不会发生吧
                t=0
            else:
                t=2.77
                n = max(prevalence1, prevalence2)
                raise Exception("有phi等于-1、1 ，n = max(prevalence1, prevalence2)值为：", n)
            if ((n>1000 and phi_ij>0 and t>=2.58) or (n>500 and phi_ij>0 and t>=2.59) or (n>200 and phi_ij>0 and t>=2.60) or (n>90 and phi_ij>0 and t>=2.63) or (n>80 and phi_ij>0 and t>=2.64) or (n>70 and phi_ij>0 and t>=2.65) or (n>60 and phi_ij>0 and t>=2.66) or (n>50 and phi_ij>0 and t>=2.68) or (n>40 and phi_ij>0 and t>=2.70) or (n>38 and phi_ij>0 and t>=2.71) or (n>35 and phi_ij>0 and t>=2.72) or (n>33 and phi_ij>0 and t>=2.73) or (n>31 and phi_ij>0 and t>=2.74) or (n>30 and phi_ij>0 and t>=2.75) or (n>28 and phi_ij>0 and t>=2.76) or (n>27 and phi_ij>0 and t>=2.77) ):#这里只考虑了两个节点联系比随机情况下更强的情况
                edge_list.append([list_cols_name[i],list_cols_name[j],prevalence1,prevalence2,phi_ij,t,-999,Cij[i][j]])
                # 添加有意义的边到边列表中，[[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的phi值，t值，无意义位，同时患两种病的人数],....]
    print("phi，有意义的边数：",len(edge_list))
    return edge_list


def compute_CCxy_significated(Cij,prevalence,N,dic_cols):
    '''
    计算CCxy值及t值 ,99%的置信水平
    输入参数：Cij:上三角矩阵，由函数compute_Cij()计算所得；prevalence:字典，由函数choose_diseases_based_on_matrix()计算可得；N:纳入的总人数；dic_cols：疾病-稀疏矩阵列的映射
    返回值：有意义的边组成的列表edge_list
    返回列表的结构：[[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的CCxy值，t值，无意义位，同时患两种病的人数],....]
    '''
    edge_list=[]
    list_cols_name=list(dic_cols.keys())
    for i in range(Cij.shape[0]):
        for j in range(i+1,Cij.shape[0]):
            prevalence1=prevalence[list_cols_name[i]]
            prevalence2 = prevalence[list_cols_name[j]]
            CCxy=(0.1*Cij[i][j]*math.sqrt(2))/math.sqrt(((0.1*prevalence1)**2)+((0.1*prevalence2)**2))

            t=0  #初始化t
            n=0
            if CCxy < 0:
                raise Exception("有CCxy溢出啦", CCxy)
                t = 0
            elif CCxy < 1:  # CCxy=1时，会发生除零错误,|CCxy|>1时，会发生计算错误
                n = max(prevalence1, prevalence2)
                # n=N
                if n>1:
                    t = (CCxy * math.sqrt(n - 2)) / math.sqrt(1 - (CCxy ** 2))
                else:
                    t=0
            elif CCxy == 1:
                #若CCxy=1，只能是这种情况：对任何一个人，必定同时患A病和B病，且A病和B病不单独出现，这时的CCxy=1；因为前面步骤去除了流行度小于1%的疾病，所以这种情况基本不会发生吧
                t=0
                n = max(prevalence1, prevalence2)
                print("有CCxy等于1", "n= max(prevalence1, prevalence2)值为：", n)
            else:
                print("******  有CCxy大于等于1？", "CCxy值为： ****** ", CCxy)
            if ((n>1000 and t>=2.58) or (n>500 and t>=2.59) or (n>200 and t>=2.60) or (n>90 and t>=2.63) or (n>80 and t>=2.64) or (n>70  and t>=2.65) or (n>60 and t>=2.66) or (n>50 and t>=2.68) or (n>40 and t>=2.70) or (n>38 and t>=2.71) or (n>35 and t>=2.72) or (n>33 and t>=2.73) or (n>31 and t>=2.74) or (n>30 and t>=2.75) or (n>28 and t>=2.76) or (n>27 and t>=2.77)
                    or (n>26 and t>=2.78) or (n>25 and t>=2.79) or (n>24 and t>=2.80) or (n>23 and t>=2.81) or (n>22 and t>=2.82) or (n>21  and t>=2.83) or (n>20 and t>=2.85) or (n>19 and t>=2.86) or (n>18 and t>=2.88) or (n>17 and t>=2.90) or (n>16 and t>=2.92) or (n>15 and t>=2.95) or (n>14 and t>=2.98) or (n>13 and t>=3.01) or (n>12 and t>=3.06) or (n>11 and t>=3.11) or (n>10 and t>=3.17) or (n>9 and t>=3.25) or (n>8 and t>=3.36) or (n>7 and t>=3.50) or (n>6 and t>=3.71) or (n>5 and t>=4.03) or (n>4 and t>=4.60) or (n>3 and t>=5.84) or (n>2 and t>=9.93) or (n>1 and t>=63.66)):#这里只考虑了两个节点联系比随机情况下更强的情况
                edge_list.append([list_cols_name[i],list_cols_name[j],prevalence1,prevalence2,CCxy,t,-999,Cij[i][j]])
                #添加有意义的边到边列表中，[[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的phi值，t值，无意义位，同时患两种病的人数],....]
    print("CCxy，有意义的边数:",len(edge_list))
    return edge_list



def construct_network_graph(type, percentile, dic_disease_prevalence_rate, edge_list):
    '''功能：构网构图
    输入参数：percentile 分位数，只画出相关系数在percentile以上的边
    输入参数：type: type='RR': RR；type='phi':phi；type='CC':CC
    '''
    # node_name_list = sorted(list(dic_disease_prevalence_rate.keys()))
    # edge_list_RR的结构：[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的RR值，置信下区间，置信上区间, 同时患两种疾病的人数], ....]

    print("the num of edge:",len(edge_list))
    node_set=set()

    quantile_value=pd.Series([edge[4] for edge in edge_list]).quantile(percentile)
    edge_list=[i for i in edge_list if i[4]>=quantile_value]

    for edge in edge_list:
        node_set.add(edge[0])
        node_set.add(edge[1])
    node_name_list=sorted(list(node_set))  #节点名称，排序后
    print("the num of node:", len(node_name_list))
    prevalence_rate=[dic_disease_prevalence_rate[i] for i in node_name_list]  #节点对应的流行率

    g=Graph()
    g.add_vertices(len(node_name_list))
    g.vs['name']=node_name_list
    g.vs['label']=node_name_list
    g.vs['prevalence']=prevalence_rate
    g.add_edges((edge[0],edge[1]) for edge in edge_list)
    RR_list=[0 for j in range(len(edge_list))]
    if type == 'RR':
        CI_high=[0 for  j in range(len(edge_list))]
        CI_low=[0 for j in range(len(edge_list))]
    for edge in edge_list:
        edge_id=g.get_eid(edge[0],edge[1])
        RR_list[edge_id]=edge[4]
        if type == 'RR':
            CI_high[edge_id] = edge[6]
            CI_low[edge_id] = edge[5]
    g.es['weight'] = RR_list
    if type=='RR':
        g.es['RR_CI_high']=CI_high
        g.es['RR_CI_low']=CI_low
    print(summary(g))
    # g.write(save_dir+"/gml_dir/RR_Graph_all.gml","gml")
    # plot(g)
    return g


def get_diseases(df_group):
    d_set=set()
    for i in df_group['ALL_DISEASE']:
        d_set|=i
    return ", ".join(list(d_set))

if __name__ == '__main__':
    """ 
    读取数据，划分出60%用于构建疾病网络；
    并构建疾病网络 
    """

    process=[1,2,3,4,5,6]

    if 1 in process:
        table=pd.read_csv('E:/UESTC_yang/999 CRC HAs/has29610.csv',encoding='gbk')
        jbdm=['jbdm'+str(i) for i in range(1,16)]
        jbdm.append('jbdm')
        # print(jbdm)
        for c in jbdm:
            table[c]=table[c].apply(lambda x: '-' if pd.isna(x) else x)
        table['ALL_DISEASE']=table['jbdm']+', '+table['jbdm1']+', '+table['jbdm2']+', '+table['jbdm3']+', '+table['jbdm4']+', '+table['jbdm5']+', '+ table['jbdm6'] + ', ' + table['jbdm7']+', ' + table['jbdm8'] + ', ' + table['jbdm9'] +', '+ table['jbdm10'] + ', ' + table['jbdm11'] + ', ' +  table['jbdm12'] + ', ' + table['jbdm13'] +', '+ table['jbdm14'] + ', ' + table['jbdm15']
        # print(table)

        # 洗疾病
        table['ALL_DISEASE'] = table['ALL_DISEASE'].apply(lambda x: set([i[:3] for i in x.split(", ") if
                                                                         len(i) > 2 and i[0] >= 'A' and i[0] <= 'Z' and
                                                                         i[1] >= '0' and i[1] <= '9' and i[2] >= '0' and
                                                                         i[2] <= '9']))
        print(table)

        # groupby disease
        tmp=[set(i.split(', ')) for i in table.groupby('sfzh', as_index=True).apply(get_diseases).values]
        # print(tmp)
        # print(list(zip([i for i in range(len(tmp))], tmp)))
        df_p_d=pd.DataFrame(list(zip([i for i in range(len(tmp))], tmp)),columns=['SFZH','ALL_DISEASE'])

    if 2 in process:
        # 获取patient - disease矩阵（np.array）
        df_p_d, dict_all_d = get_onehot_disea(df_p_d)
        p_d_nda = np.array([i for i in df_p_d['lst_disea_binary']])
        row_ids = df_p_d[config.SFZH].values
        print('1. p_d_nda.shape,len(dict_all_d)', p_d_nda.shape, len(dict_all_d))    # (29610, 1429) 1429

    if 3 in process:
        # 去除快病，构建患者-疾病稀疏矩阵
        p_d_nda, dict_selected_d = construct_sparse_matrix(p_d_nda, dict_all_d, is_chronic=1)   # is_chronic=0表示不去除急性病
        print('2, p_d_nda.shape,len(dict_selected_d)', p_d_nda.shape, len(dict_selected_d))  # (29610, 621) 621

    if 4 in process:
        # 去除流行度小于k%的疾病，更新患者-慢病稀疏矩阵
        prevalence_threshold = 0.01
        has_research_d = 0
        lst_research_d = config.CRC
        csc_matrix_final, dic_cols_new, dic_prevalence, dic_disease_prevalence_rate, num_patient \
            = choose_diseases_based_on_matrix(p_d_nda, dict_selected_d, has_research_d, prevalence_threshold, lst_research_d)
        # 去除流行率小于1 % 的疾病后，稀疏矩阵的维度： (29610, 97)
        # 去除流行率小于0.00100的疾病后，稀疏矩阵的维度： (29610, 271)


    if 5 in process:
        # 生成共慢病稀疏矩阵，计算Cij
        Cij = compute_Cij(dic_cols_new, csc_matrix_final)
        # print(list(Cij))
        print('Cij.shape',Cij.shape)

        edge_type = ['phi', 'RR', 'CC'][1]
        # 计算phi, 选取有意义的边
        if edge_type == ['phi', 'RR', 'CC'][0]:
            edge_list = compute_phi_significated(Cij, dic_prevalence, dic_cols_new, num_patient)
        elif edge_type == ['phi', 'RR', 'CC'][1]:
            edge_list = compute_RR_CI(Cij, dic_prevalence,num_patient, dic_cols_new)
        elif edge_type == ['phi', 'RR', 'CC'][2]:
            edge_list =compute_CCxy_significated(Cij, dic_prevalence, num_patient, dic_cols_new)
        pd.DataFrame(edge_list).to_csv('edge_list_%f.csv'%(prevalence_threshold),index=False)
    if 6 in process:
        # # 构网，画图
        percentile=0
        g=construct_network_graph(edge_type, percentile, dic_disease_prevalence_rate, edge_list)
        g.write("RR_DCN_%f.gml"%(prevalence_threshold),"gml")


