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



class FeatureEmbedder(object):
  """This class is used to convert SparseTensor inputs to dense Tensors.

  This class is used to convert raw features to their vector representations.
  It takes in a dictionary, where the key is the name of a feature (e.g.
  diagnosis_id) and the value is a SparseTensor of integers (i.e. lookup IDs),
  then retrieves corresponding vector representations using tf.embedding.lookup.
  """

  def __init__(self, vocab_sizes, feature_keys, embedding_size,init_dis_emd):
    """Init function.

    Args:
      vocab_sizes: A dictionary of vocabularize sizes for each feature.
      feature_keys: A list of feature names you want to use.
      embedding_size: The dimension size of the feature representation vector.
    """
    self._params = {}
    self._feature_keys = feature_keys
    self._vocab_sizes = vocab_sizes
    dummy_emb = tf.zeros([1, embedding_size], dtype=tf.float32)

    for feature_key in feature_keys:
      vocab_size = self._vocab_sizes[feature_key]
      if feature_key=='dx_ints' and init_dis_emd.shape[0]==vocab_size:
          print('====yes====')
          emb = tf.get_variable(feature_key,
                                shape=(vocab_size, embedding_size),
                                dtype=tf.float32,
                                initializer = tf.constant_initializer(init_dis_emd))   # dx_ints: (3249, embedding_size)
      else:
          emb = tf.get_variable(feature_key,
                                shape=(vocab_size, embedding_size),
                                dtype=tf.float32)
      self._params[feature_key] = tf.concat([emb, dummy_emb], axis=0)

    self._params['visit'] = tf.get_variable('visit', shape=(1, embedding_size), dtype=tf.float32)

  def lookup(self, dic_features, dic_masks):
    """Look-up function.

    This function converts the SparseTensor of integers to a dense Tensor of
    tf.float32.

    Args:
      feature_map: A dictionary of SparseTensors for each feature.
      max_num_codes: The maximum number of how many feature there can be inside
        a single visit, per feature. For example, if this is set to 50, then we
        are assuming there can be up to 50 diagnosis codes, 50 treatment codes,
        and 50 lab codes. This will be used for creating the prior matrix.

    Returns:
      embeddings: A dictionary of dense representation Tensors for each feature.
      masks: A dictionary of dense float32 Tensors for each feature, that will
        be used as a mask in the downstream tasks.
    """
    # dic_features={'dx_ints':feat_dx_ints}
    # dic_masks={'dx_ints':mask_dx_ints}

    masks = {}
    embeddings = {}
    for key in self._feature_keys:
      # if max_num_codes > 0:
      #   feature = tf.SparseTensor(indices=feature_map[key].indices, values=feature_map[key].values,
      #       dense_shape=[feature_map[key].dense_shape[0], feature_map[key].dense_shape[1], max_num_codes])   # 将batch中每个患者的dx数和proc数最大值设置为max_num_codes
      #   # feature.shape  =  [batch size, 1, max_num_codes]
      # else:
      #   feature = feature_map[key]
      # feature_ids = tf.sparse.to_dense(feature, default_value=self._vocab_sizes[key])
      # # with tf.Session() as sess:
      # #   feature_ids=sess.run(feature_ids)
      # #   print('feature_ids',feature_ids.shape, feature_ids)
      # #  # feature_ids  (32, 1, 50)    [[[89   13   22... 3249 3249 3249]]       [[5   13   59... 3249 3249 3249]]       [[89   15  308... 3249 3249 3249]] ......]
      # #  #  {'dx_ints':3249, 'proc_ints':2210}
      #
      # # 初始化embedding
      # feature_ids = tf.squeeze(feature_ids, axis=1)  #  the shape of feature_ids : (32, 50)
      embeddings[key] = tf.nn.embedding_lookup(self._params[key], dic_features[key])   # self._params['dx_ints']: 维度为(3249+1<dummy_emb>, embedding_size)的可训练参数
      # the shape of embeddings['dx_ints'] :   (batch_size, 50, embedding_size)

      # # 初始化mask
      # mask = tf.SparseTensor(indices=feature.indices, values=tf.ones(tf.shape(feature.values)), dense_shape=feature.dense_shape)
      # masks[key] = tf.squeeze(tf.sparse.to_dense(mask), axis=1)  #  the shape of masks['dx_ints']: (32, 50)
      masks[key]=dic_masks[key]

    batch_size = tf.shape(list(embeddings.values())[0])[0]    # batch_size=32
    # with tf.Session() as sess:
    #   sess.run(tf.global_variables_initializer())
    #   print('batch_size', sess.run(batch_size))

    embeddings['visit'] = tf.tile(self._params['visit'][None, :, :], [batch_size, 1, 1])
    #  tf.tile(输入, 同一维度上复制的次数)：平铺，用于在同一维度上的复制。 输入——self._params['visit'] : (1, embedding_size)
    #  输出 embeddings['visit'] : (batch_size, 1, embedding_size)   每一个embedding（初始化时）是一样的
    masks['visit'] = tf.ones(batch_size,dtype=tf.int32)[:, None]    # (32, 1)
    # with tf.Session() as sess:
    #   sess.run(tf.global_variables_initializer())
    #   print('1. masks',sess.run(masks['dx_ints']).shape)

    # return:
    # embeddings.keys(): ['dx_ints', 'proc_ints', 'visit]
    # embeddings['dx_ints'] and  embeddings['proc_ints'] : (batch_size, 50, embedding_size); embeddings['visit'] : (batch_size, 1, embedding_size) - 初始化时向量相同;
    # masks['dx_ints'] and masks['proc_ints'] : shape-(batch_size, 50), value-【[0，疾病个数)为1, [疾病个数, 50]为0】;  masks['visit'] : (batch_size, 1) , 全1;
    return embeddings, masks


class GraphConvolutionalTransformer(object):
  """Graph Convolutional Transformer class.

  This is an implementation of Graph Convolutional Transformer. With a proper
  set of options, it can be used as a vanilla Transformer.
  """

  def __init__(self,
               embedding_size=128,
               num_transformer_stack=3,
               num_feedforward=2,
               num_attention_heads=1,
               ffn_dropout=0.1,
               attention_normalizer='softmax',
               multihead_attention_aggregation='concat',
               directed_attention=False,
               use_inf_mask=True,
               use_prior=True):
    """Init function.

    Args:
      embedding_size: The size of the dimension for hidden layers.
      num_transformer_stack: The number of Transformer blocks.
      num_feedforward: The number of layers in the feedforward part of
        Transformer.
      num_attention_heads: The number of attention heads.
      ffn_dropout: Dropout rate used inside the feedforward part.
      attention_normalizer: Use either 'softmax' or 'sigmoid' to normalize the
        attention values.
      multihead_attention_aggregation: Use either 'concat' or 'sum' to handle
        the outputs from multiple attention heads.
      directed_attention: Decide whether you want to use the unidirectional
        attention, where information accumulates inside the dummy visit node.
      use_inf_mask: Decide whether you want to use the guide matrix. Currently
        unused.
      use_prior: Decide whether you want to use the conditional probablility
        information. Currently unused.
      **kwargs: Other arguments to tf.keras.layers.Layer init.
    """

    super(GraphConvolutionalTransformer, self).__init__()
    self._hidden_size = embedding_size
    self._num_stack = num_transformer_stack
    self._num_feedforward = num_feedforward
    self._num_heads = num_attention_heads
    self._ffn_dropout = ffn_dropout
    self._attention_normalizer = attention_normalizer
    self._multihead_aggregation = multihead_attention_aggregation
    self._directed_attention = directed_attention
    self._use_inf_mask = use_inf_mask
    self._use_prior = use_prior

    self._layers = {}
    self._layers['Q'] = []
    self._layers['K'] = []
    self._layers['V'] = []
    self._layers['ffn'] = []
    self._layers['head_agg'] = []

    for i in range(self._num_stack):
      self._layers['Q'].append(tf.keras.layers.Dense(self._hidden_size * self._num_heads, use_bias=False))
      self._layers['K'].append(tf.keras.layers.Dense( self._hidden_size * self._num_heads, use_bias=False))
      self._layers['V'].append(tf.keras.layers.Dense(self._hidden_size * self._num_heads, use_bias=False))

      if self._multihead_aggregation == 'concat':
        self._layers['head_agg'].append(tf.keras.layers.Dense(self._hidden_size, use_bias=False))

      self._layers['ffn'].append([])
      # Don't need relu for the last feedforward.
      for _ in range(self._num_feedforward - 1):
        self._layers['ffn'][i].append(tf.keras.layers.Dense(self._hidden_size, activation='relu'))
      self._layers['ffn'][i].append(tf.keras.layers.Dense(self._hidden_size))

  def feedforward(self, features, stack_index, keep_prob_=1.):
    """Feedforward component of Transformer.

    Args:
      features: 3D float Tensor of size (batch_size, num_features,
        embedding_size). This is the input embedding to GCT.
      stack_index: An integer to indicate which Transformer block we are in.
      training: Whether to run in training or eval mode.

    Returns:
      Latent representations derived from this feedforward network.
    """
    for i in range(self._num_feedforward):
      features = self._layers['ffn'][stack_index][i](features)
      # if training:
      features = tf.nn.dropout(features, keep_prob=keep_prob_)
      # else:
      #   features = tf.nn.dropout(features, rate=1.)

    return features

  def qk_op(self,
            features,
            stack_index,
            batch_size,
            num_codes,
            attention_mask,
            inf_mask=None,
            directed_mask=None):
    """Attention generation part of Transformer.

    Args:
      features: 3D float Tensor of size (batch_size, num_features,
        embedding_size). This is the input embedding to GCT.
      stack_index: An integer to indicate which Transformer block we are in.
      batch_size: The size of the mini batch.
      num_codes: The number of features (i.e. codes) given as input.
      attention_mask: A Tensor for suppressing the attention on the padded
        tokens.
      inf_mask: The guide matrix to suppress the attention values to zeros for
        certain parts of the attention matrix (e.g. diagnosis codes cannot
        attend to other diagnosis codes).
      directed_mask: If the user wants to only use the upper-triangle of the
        attention for uni-directional attention flow, we use this strictly lower
        triangular matrix filled with infinity.

    Returns:
      The attention distribution derived from the QK operation.
    """

    # features：(batch_size, 1+50+50, embedding_size)    # 第二维中，是通过padding统一到1+50+50维的；padding行的embedding为全0
    # attention_mask： 将mask中原来填充的位置（0的位置）设置为正无穷大，其他位置为设置为0; （batch_size, 101）
    # inf_mask：将guide中原来0的位置设置为正无穷大，其他位置为设置为0;   (batch_size, 1+max_num_codes*2, 1+max_num_codes*2)
    # directed_mask：下三角阵为无穷大，diag=0，上三角全为0;  (1, 1, 101, 101)


    q = self._layers['Q'][stack_index](features)    # tf.keras.layers.Dense(self._hidden_size * self._num_heads, use_bias=False)
    # shape of q: (batch_size, 1+50+50, self._hidden_size * self._num_heads)
    q = tf.reshape(q, [batch_size, num_codes, self._hidden_size, self._num_heads])

    k = self._layers['K'][stack_index](features)
    k = tf.reshape(k, [batch_size, num_codes, self._hidden_size, self._num_heads])

    # Need to transpose q and k to (2, 0, 1)
    q = tf.transpose(q, perm=[0, 3, 1, 2])
    k = tf.transpose(k, perm=[0, 3, 2, 1])
    pre_softmax = tf.matmul(q, k) / tf.sqrt(tf.cast(self._hidden_size, tf.float32))
    # shape of pre_softmax: (batch_size, _num_heads, 1+50+50, 1+50+50)

    # attention_mask： 将mask中原来填充的位置（0的位置）设置为正无穷大，其他位置为设置为0; （batch_size, 101）
    pre_softmax -= attention_mask[:, None, None, :]

    if inf_mask is not None:
      # inf_mask：将guide中原来0的位置设置为正无穷大，其他位置为设置为0;   (batch_size, 101, 101)
      pre_softmax -= inf_mask[:, None, :, :]

    if directed_mask is not None:
      # directed_mask：下三角阵为无穷大，diag=0，上三角全为0;  (1, 1, 101, 101)
      pre_softmax -= directed_mask

    if self._attention_normalizer == 'softmax':
      attention = tf.nn.softmax(pre_softmax, axis=3)
    else:
      attention = tf.nn.sigmoid(pre_softmax)

    # with tf.Session() as sess:
    #   sess.run(tf.global_variables_initializer())
    #   print('------pre_softmax', attention_mask.shape, sess.run(pre_softmax))
    #   print('------attention', sess.run(attention))
    return attention    # (batch_size, _num_heads, 1+50+50, 1+50+50)   对dim=3做softmax；


  def get_emb_atte(self, features, masks, guide=None, prior_guide=None,keep_prob_=0.1):
    """This function transforms the input embeddings.

    This function converts the SparseTensor of integers to a dense Tensor of tf.float32.

    Args:
      features: 3D float Tensor of size (batch_size, num_features,
        embedding_size). This is the input embedding to GCT.
      masks: 3D float Tensor of size (batch_size, num_features, 1). This holds
        binary values to indicate which parts are padded and which are not.
      guide: 3D float Tensor of size (batch_size, num_features, num_features).
        This is the guide matrix.
      prior_guide: 3D float Tensor of size (batch_size, num_features,
        num_features). This is the conditional probability matrix.
      training: Whether to run in training or eval mode.

    Returns:
      features: The final layer of GCT.
      attentions: List of attention values from all layers of GCT. This will be
        used later to regularize the self-attention process.
    """

    # ================================
    # features (传入参数embeddings): (batch_size, 1+50+50, embedding_size)    # 第二维中，是通过padding统一到1+50+50维的
    # masks (传入参数masks[:, :, None]): (batch_size, 1+50+50, 1)
    # guide: （batch_size, 1+max_num_codes*2, 1+max_num_codes*2）；guide是自身存在的特征间关系矩阵（0或1）
    # prior_guide: （batch_size, 1+max_num_codes*2, 1+max_num_codes*2） ；prior_guide是特征间条件概率矩阵（0到1），对dim=2归一化了
    # ================================

    batch_size = tf.shape(features)[0]
    num_codes = tf.shape(features)[1]   # 101

    # Use the given masks to create a negative infinity Tensor to suppress the
    # attention weights of the padded tokens. Note that the given masks has
    # the shape (batch_size, num_codes, 1), so we remove the last dimension
    # during the process.

    mask_idx = tf.cast(tf.where(tf.equal(masks[:, :, 0], 0)), tf.int32)  # (2705, 2) [[  0   6] [  0   7] [  0   8] ... [ 31  98] [ 31  99] [ 31 100]]
    mask_matrix = tf.fill([tf.shape(mask_idx)[0]], tf.float32.max)   # [tf.shape(mask_idx)[0]] [2705]
    # mask_matrix : (2705,) [3.4028235e+38 3.4028235e+38 3.4028235e+38 ... 3.4028235e+38 3.4028235e+38]
    attention_mask = tf.scatter_nd(indices=mask_idx, updates=mask_matrix, shape=tf.shape(masks[:, :, 0]))
    # attention_mask： 将mask中原来填充的位置（0的位置）设置为正无穷大，其他位置为设置为0; （batch_size, 101）

    # with tf.Session() as sess:
    #   sess.run(tf.global_variables_initializer())
    #   print('------attention_mask',attention_mask.shape, sess.run(attention_mask))

    inf_mask = None
    if self._use_inf_mask:
      guide_idx = tf.cast(tf.where(tf.equal(guide, 0.)), tf.int32)
      inf_matrix = tf.fill([tf.shape(guide_idx)[0]], tf.float32.max)
      inf_mask = tf.scatter_nd(indices=guide_idx, updates=inf_matrix, shape=tf.shape(guide))
      # inf_mask： 将guide中原来0的位置设置为正无穷大，其他位置为设置为0;   (batch_size, 1+max_num_codes*2, 1+max_num_codes*2)

    directed_mask = None
    if self._directed_attention:   # 论文中设置为Fasle
      inf_matrix = tf.fill([num_codes, num_codes], tf.float32.max)             # (101, 101)
      inf_matrix = tf.matrix_set_diag(inf_matrix, tf.zeros(num_codes))
      directed_mask = tf.matrix_band_part(inf_matrix, -1, 0)[None, None, :, :]    # 取下三角阵(包括diag)，上三角全为0; (1, 1, 101, 101)

    attention = None
    attentions = []

    attention=tf.tile(prior_guide[:, None, :, :], [1, self._num_heads, 1, 1])
    attention = attention * masks[:, :, 0][:, None, :, None]
    attentions.append(attention)

    for i in range(self._num_stack):   # Transfromer的层数
      features = masks * features   # 使padding的行（embedding）为全0

      if self._use_prior and i <=5:    # 第一层用prior_duide
        # attention = tf.tile(prior_guide[:, None, :, :],  [1, self._num_heads, 1, 1])      # （batch_size, num_heads, 1+max_num_codes*2, 1+max_num_codes*2）
        # prior_guide[:, None, :, :]: （batch_size, 1, 1+max_num_codes*2, 1+max_num_codes*2）  #第2个维度是注意力头数
        attention = self.qk_op(features, i, batch_size, num_codes, attention_mask, inf_mask, directed_mask)

      else:      # 第二层开始用自注意力
        # （每层的自注意力都是基于原始features embedding计算的）    # softmax得到的attention
        attention = self.qk_op(features, i, batch_size, num_codes, attention_mask, inf_mask, directed_mask)
        # attention： (batch_size, _num_heads, 1+50+50, 1+50+50)   对dim=3做softmax；

      attention=attention*masks[:,:,0][:,None,:,None]
      attentions.append(attention)    # 收集每一层的attention

      v = self._layers['V'][i](features)    # tf.keras.layers.Dense(self._hidden_size * self._num_heads, use_bias=False)
      # shape of v: (batch_size, 1+50+50, self._hidden_size * self._num_heads)
      v = tf.reshape(v, [batch_size, num_codes, self._hidden_size, self._num_heads])
      v = tf.transpose(v, perm=[0, 3, 1, 2])    #  [batch_size, num_heads, num_codes, hidden_size]
      # post_attention is (batch, num_heads, num_codes, hidden_size)
      post_attention = tf.matmul(attention, v)

      if self._num_heads == 1:
        post_attention = tf.squeeze(post_attention, axis=1)
      elif self._multihead_aggregation == 'concat':
        # post_attention is (batch, num_codes, num_heads, hidden_size)
        post_attention = tf.transpose(post_attention, perm=[0, 2, 1, 3])
        # post_attention is (batch, num_codes, num_heads*hidden_size)
        post_attention = tf.reshape(post_attention, [batch_size, num_codes, self._num_heads*self._hidden_size])
        # print('2. post_attention',post_attention)
        # post attention is (batch, num_codes, hidden_size)
        post_attention = self._layers['head_agg'][i](post_attention)
      else:
        post_attention = tf.reduce_sum(post_attention, axis=1)

      # Residual connection + layer normalization
      post_attention += features
      post_attention = tf.contrib.layers.layer_norm(post_attention, begin_norm_axis=2)

      # Feedforward component + residual connection + layer normalization
      post_ffn = self.feedforward(post_attention, i, keep_prob_)
      post_ffn += post_attention
      post_ffn = tf.contrib.layers.layer_norm(post_ffn, begin_norm_axis=2)

      features = post_ffn

    # return
    # features: (batch, num_codes, embedding_size)
    # attention： (batch_size, _num_heads, 1+50+50, 1+50+50)   对dim=3做softmax；
    return features * masks, attentions


def create_matrix_dx_cx(df_disease_train):
    #   df_disease_train:
    #   ['patient_id',
    #   'ls_dx_names', 'ls_dx_ints', 'dx_prior_indices', 'dx_prior_values','ls_dx_ints_imputed','mask','guide','prior_guide',
    #  'cx_bucket', 'cx_bucket_idx', 'cx_prior_indices', 'cx_prior_values', 'ls_cx_ints_imputed', 'cx_mask',  'guide_cx',  'prior_guide_cx',
    # todo  'guide_dx_cx',  'prior_guide_dx_cx']

    # 构建guide_dx_cx

    # 由dx和cx的矩阵拼接而成
    dx_guide=np.array([i for i in df_disease_train['guide']])
    cx_guide=np.array([i for i in df_disease_train['guide_cx']])
    dx_guide = tf.convert_to_tensor(dx_guide, dtype=tf.float32)   #(batch size, 31, 31)
    cx_guide = tf.convert_to_tensor(cx_guide, dtype=tf.float32)    #(batch size, 68, 68)

    batch_size=dx_guide.shape[0]
    dx_len=dx_guide.shape[-1]-1
    cx_len=cx_guide.shape[-1]-1

    ## 把cx_guide拆掉
    a = tf.gather(cx_guide, indices=list(range(1, cx_guide.shape[-1])), axis=-1)    # (batch size, 68, 67)
    col3_1 = tf.gather(a, indices=[0], axis=-2)    # (batch size, 1, 67)
    col3_3 = tf.gather(a, indices=list(range(1, a.shape[-2])), axis=-2)    # (batch size, 67, 67)
    b = tf.gather(cx_guide, indices=[0], axis=-1)  # (batch size, 68, 1)
    col1_3=tf.gather(b, indices=list(range(1, a.shape[-2])), axis=-2)    # (batch size, 67, 1)
    ## 新建0矩阵
    col3_2=tf.tile(tf.zeros([dx_len, cx_len])[None, :, :], [batch_size, 1, 1])    # (batch size, 30, 67)
    col2_3=tf.tile(tf.zeros([cx_len, dx_len])[None, :, :], [batch_size, 1, 1])    # (batch size, 67, 30)
    ## 拼
    tmp = tf.concat([col3_1, col3_2], axis=1)     # (batch size, 31, 67)
    guide_dx_cx = tf.concat([dx_guide, tmp], axis=-1)      # (batch size, 31, 31+67)
    tmp= tf.concat([col1_3,col2_3, col3_3], axis=-1)       # (batch size, 67, 1+30+67)

    guide_dx_cx = tf.concat([guide_dx_cx, tmp], axis=1)      # (batch size, 31+67, 31+67)


    # 构建prior_guide_dx_cx
    dx_prior_guide = np.array([i for i in df_disease_train['prior_guide']])
    cx_prior_guide = np.array([i for i in df_disease_train['prior_guide_cx']])
    dx_prior_guide = tf.convert_to_tensor(dx_prior_guide, dtype=tf.float32)  # (batch size, 31, 31)
    cx_prior_guide = tf.convert_to_tensor(cx_prior_guide, dtype=tf.float32)  # (batch size, 68, 68)

    batch_size = dx_prior_guide.shape[0]
    dx_len = dx_prior_guide.shape[-1] - 1
    cx_len = cx_prior_guide.shape[-1] - 1

    a = tf.gather(cx_prior_guide, indices=list(range(1, cx_prior_guide.shape[-1])), axis=-1)  # (batch size, 68, 67)
    col3_1 = tf.gather(a, indices=[0], axis=-2)  # (batch size, 1, 67)
    col3_3 = tf.gather(a, indices=list(range(1, a.shape[-2])), axis=-2)  # (batch size, 67, 67)
    b = tf.gather(cx_prior_guide, indices=[0], axis=-1)  # (batch size, 68, 1)
    col1_3 = tf.gather(b, indices=list(range(1, a.shape[-2])), axis=-2)  # (batch size, 67, 1)
    ## 新建0矩阵
    col3_2=tf.tile(tf.zeros([dx_len, cx_len])[None, :, :], [batch_size, 1, 1])    # (batch size, 30, 67)
    col2_3=tf.tile(tf.zeros([cx_len, dx_len])[None, :, :], [batch_size, 1, 1])    # (batch size, 67, 30)
    ## 拼
    tmp = tf.concat([col3_1, col3_2], axis=1)     # (batch size, 31, 67)
    prior_guide_dx_cx = tf.concat([dx_prior_guide, tmp], axis=-1)      # (batch size, 31, 31+67)
    tmp= tf.concat([col1_3,col2_3, col3_3], axis=-1)       # (batch size, 67, 1+30+67)
    prior_guide_dx_cx = tf.concat([prior_guide_dx_cx, tmp], axis=1)      # (batch size, 31+67, 31+67)

    # with tf.Session() as sess:
    sess = tf.Session()
    guide_dx_cx = sess.run(guide_dx_cx)
    prior_guide_dx_cx = sess.run(prior_guide_dx_cx)

    df_disease_train['guide_dx_cx'] = [np.array(i) for i in guide_dx_cx]
    df_disease_train['prior_guide_dx_cx'] = [np.array(i) for i in prior_guide_dx_cx]

    return df_disease_train   # guide是自身存在的特征间关系矩阵（0或1）；prior_guide是特征间条件概率矩阵（0到1）；（batch_size, 1+max_num_codes*2, 1+max_num_codes*2）


def create_ps_matrix_only_dx(df_disease_train, use_prior, use_inf_mask, max_num_codes, prior_scalar):
  """Creates guide matrix and prior matrix when feature_set='vdp'.

  This function creates the guide matrix and the prior matrix when visits
  include diagnosis codes, treatment codes, but not lab codes.

  Args:
    features: A dictionary of SparseTensors for each feature.
    mask: 3D float Tensor of size (batch_size, num_features, 1). This holds
      binary values to indicate which parts are padded and which are not.
    use_prior: Whether to create the prior matrix.
    use_inf_mask : Whether to create the guide matrix.
    max_num_codes: The maximum number of how many feature there can be inside a
      single visit, per feature. For example, if this is set to 50, then we are
      assuming there can be up to 50 diagnosis codes and 50 treatment codes.
      This will be used for creating the prior matrix.
    prior_scalar: A float value between 0.0 and 1.0 to be used to hard-code the
      diagnoal elements of the prior matrix.

  Returns:
    guide: The guide matrix.
    prior_guide: The conditional probablity matrix.
  """
  # dx_ids = features['dx_ints']
  # proc_ids = features['proc_ints']

  #   df_disease_p:   ['patient_id', 'ls_dx_ints', 'ls_dx_names', 'dx_prior_indices', 'dx_prior_values','ls_dx_ints_imputed','mask']

  batch_size=df_disease_train.shape[0]
  mask=np.array([i for i in df_disease_train['mask']])
  mask=tf.convert_to_tensor(mask,dtype=tf.float32)
  mask = tf.concat([tf.ones(batch_size, dtype=tf.float32)[:, None], mask], axis=1)

  # batch_size = dx_ids.dense_shape[0]
  num_dx_ids = max_num_codes
  num_codes = 1 + num_dx_ids

  prior_guide = None
  if use_prior:

    prior_indices =[i for i in df_disease_train['dx_prior_indices']]
    prior_idx_values = [i for i in df_disease_train['dx_prior_values']]

    prior_idx_dx = []
    for p_i in range(len(prior_indices)):
        prior_idx_dx.extend([[p_i, dx_i[0], dx_i[1]] for dx_i in prior_indices[p_i]])

    prior_idx_values_dx = []
    for i in prior_idx_values:
        prior_idx_values_dx.extend([1.] * len(i))
        # prior_idx_values_dx.extend(i)
        # prior_idx_values_dx.extend([prior_scalar]*len(i))

    # 排序
    prior_idx_dx=pd.DataFrame(prior_idx_dx,columns=['p','d1','d2'])
    prior_idx_dx['v']=prior_idx_values_dx
    prior_idx_dx=prior_idx_dx.sort_values(by=['p','d1','d2'])
    prior_idx_values_dx=prior_idx_dx['v'].values
    prior_idx_dx=prior_idx_dx[['p','d1','d2']].values

    sparse_prior = tf.SparseTensor(indices=prior_idx_dx, values=prior_idx_values_dx, dense_shape=[batch_size, num_dx_ids, num_dx_ids])

    prior_guide = tf.sparse.to_dense(sparse_prior, validate_indices=True)    # dx和proc的条件概率矩阵  （32，100，100）
    # with tf.Session() as sess:
    #   print('prior_guide',sess.run(prior_guide))
    prior_guide=tf.cast(prior_guide, tf.float32)
    # 在prior_guide的基础上，添加visit对应行和列
    visit_guide = tf.convert_to_tensor( [prior_scalar] * max_num_codes ,  dtype=tf.float32)    # [0.5 0.5 0.5 0.5 0.5  0.  0.  0.  0.  0. ]
    prior_guide = tf.concat([tf.tile(visit_guide[None, None, :], [batch_size, 1, 1]), prior_guide],axis=1)
    # visit_guide （batch_size, 1, max_num_codes*2）; concat prior_guide后：（batch_size, 1+max_num_codes*2, max_num_codes*2）

    visit_guide = tf.concat([[0.0], visit_guide], axis=0)
    prior_guide = tf.concat([tf.tile(visit_guide[None, :, None], [batch_size, 1, 1]), prior_guide],axis=2)
    # visit_guide  (32, 1+max_num_codes*2, 1); concat prior_guide后：（batch_size, 1+max_num_codes*2, 1+max_num_codes*2）

    prior_guide = (prior_guide * mask[:, :, None] * mask[:, None, :] + tf.eye(num_codes)[None, :, :])   # 条件概率矩阵针对患者个体化，并添加自环
    # degrees = tf.reduce_sum(prior_guide, axis=2)
    # prior_guide = prior_guide / degrees[:, :, None]   # 按列归一化

    sess = tf.Session()
    prior_guide=sess.run(prior_guide)

  df_disease_train['prior_guide_ntw'] = [np.array(i) for i in prior_guide]
  return df_disease_train   # guide是自身存在的特征间关系矩阵（0或1）；prior_guide是特征间条件概率矩阵（0到1）；（batch_size, 1+max_num_codes*2, 1+max_num_codes*2）




def create_matrix_only_dx(df_disease_train, use_prior, use_inf_mask, max_num_codes, prior_scalar):
  """Creates guide matrix and prior matrix when feature_set='vdp'.

  This function creates the guide matrix and the prior matrix when visits
  include diagnosis codes, treatment codes, but not lab codes.

  Args:
    features: A dictionary of SparseTensors for each feature.
    mask: 3D float Tensor of size (batch_size, num_features, 1). This holds
      binary values to indicate which parts are padded and which are not.
    use_prior: Whether to create the prior matrix.
    use_inf_mask : Whether to create the guide matrix.
    max_num_codes: The maximum number of how many feature there can be inside a
      single visit, per feature. For example, if this is set to 50, then we are
      assuming there can be up to 50 diagnosis codes and 50 treatment codes.
      This will be used for creating the prior matrix.
    prior_scalar: A float value between 0.0 and 1.0 to be used to hard-code the
      diagnoal elements of the prior matrix.

  Returns:
    guide: The guide matrix.
    prior_guide: The conditional probablity matrix.
  """
  # dx_ids = features['dx_ints']
  # proc_ids = features['proc_ints']

  #   df_disease_p:   ['patient_id', 'ls_dx_ints', 'ls_dx_names', 'dx_prior_indices', 'dx_prior_values','ls_dx_ints_imputed','mask']
  # print(df_disease_train['ls_dx_ints'])
  # print(df_disease_train['dx_prior_indices'])
  # print(df_disease_train['dx_prior_values'])
  # print(df_disease_train['mask'], type(df_disease_train['mask'].values[0]))

  batch_size=df_disease_train.shape[0]
  mask=np.array([i for i in df_disease_train['mask']])
  mask=tf.convert_to_tensor(mask,dtype=tf.float32)
  mask = tf.concat([tf.ones(batch_size, dtype=tf.float32)[:, None], mask], axis=1)

  # batch_size = dx_ids.dense_shape[0]
  num_dx_ids = max_num_codes
  num_codes = 1 + num_dx_ids

  guide = None
  if use_inf_mask:
    row0 = tf.concat([tf.ones([1, 1]), tf.ones([1, num_dx_ids])], axis=1)   # (1, 101)  [[0, 1,1,1,1,....,1,0,0,0,0,....,0]]
    row1 = tf.concat([tf.ones([num_dx_ids, 1 + num_dx_ids])],  axis=1)   # (num_dx_ids, 1+num_dx_ids+num_proc_ids)
    # row2 = tf.zeros([num_proc_ids, num_codes])

    guide = tf.concat([row0, row1], axis=0)   # (101,101)
    guide=guide - tf.eye(num_codes)[ :, :]
    # guide = guide + tf.transpose(guide)
    guide = tf.tile(guide[None, :, :], [batch_size, 1, 1])   # (batch_size, 101,101)
    # mask (batch_size, 1+50+50)； mask[:, :, None] (32, 101, 1) ； mask[:, None,:]  (32, 1, 101)； mask[None,:, :]   (1, 32, 101)

    guide = (guide * mask[:, :, None] * mask[:, None, :] + tf.eye(num_codes)[None, :, :])   # (batch_size, 101, 101)
    # guide: 针对每个患者个体，只有在mask中为1的dx和proc对应的行和列, 在上一步生成的guide中才继续保留为1，包含自环（对角均为1）; 每个患者的guide不一样
    # with tf.Session() as sess:
    #   print('guide', sess.run(guide))

  prior_guide = None
  if use_prior:

    prior_indices =[i for i in df_disease_train['dx_prior_indices']]
    prior_idx_values = [i for i in df_disease_train['dx_prior_values']]

    prior_idx_dx = []
    for p_i in range(len(prior_indices)):
        prior_idx_dx.extend([[p_i, dx_i[0], dx_i[1]] for dx_i in prior_indices[p_i]])

    prior_idx_values_dx = []
    for i in prior_idx_values:
        prior_idx_values_dx.extend(i)
        # prior_idx_values_dx.extend([prior_scalar]*len(i))


    # print('prior_idx_dx',prior_idx_dx)
    # print('prior_idx_values_dx',prior_idx_values_dx)
    # print('[batch_size, num_dx_ids, num_dx_ids]',[batch_size, num_dx_ids, num_dx_ids])
    # 排序
    prior_idx_dx=pd.DataFrame(prior_idx_dx,columns=['p','d1','d2'])
    prior_idx_dx['v']=prior_idx_values_dx
    prior_idx_dx=prior_idx_dx.sort_values(by=['p','d1','d2'])
    prior_idx_values_dx=prior_idx_dx['v'].values
    prior_idx_dx=prior_idx_dx[['p','d1','d2']].values

    sparse_prior = tf.SparseTensor(indices=prior_idx_dx, values=prior_idx_values_dx, dense_shape=[batch_size, num_dx_ids, num_dx_ids])

    prior_guide = tf.sparse.to_dense(sparse_prior, validate_indices=True)    # dx和proc的条件概率矩阵  （32，100，100）
    # with tf.Session() as sess:
    #   print('prior_guide',sess.run(prior_guide))
    prior_guide=tf.cast(prior_guide, tf.float32)
    # 在prior_guide的基础上，添加visit对应行和列
    visit_guide = tf.convert_to_tensor( [prior_scalar] * max_num_codes ,  dtype=tf.float32)    # [0.5 0.5 0.5 0.5 0.5  0.  0.  0.  0.  0. ]
    prior_guide = tf.concat([tf.tile(visit_guide[None, None, :], [batch_size, 1, 1]), prior_guide],axis=1)
    # visit_guide （batch_size, 1, max_num_codes*2）; concat prior_guide后：（batch_size, 1+max_num_codes*2, max_num_codes*2）

    visit_guide = tf.concat([[0.0], visit_guide], axis=0)
    prior_guide = tf.concat([tf.tile(visit_guide[None, :, None], [batch_size, 1, 1]), prior_guide],axis=2)
    # visit_guide  (32, 1+max_num_codes*2, 1); concat prior_guide后：（batch_size, 1+max_num_codes*2, 1+max_num_codes*2）

    # prior_guide = (prior_guide * mask[:, :, None] * mask[:, None, :] +prior_scalar * tf.eye(num_codes)[None, :, :])   # 条件概率矩阵针对患者个体化，并添加自环
    prior_guide = (prior_guide * mask[:, :, None] * mask[:, None, :] +  tf.eye(num_codes)[None, :, :])  # 条件概率矩阵针对患者个体化，并添加自环

    # degrees = tf.reduce_sum(prior_guide, axis=2)
    # prior_guide = prior_guide / degrees[:, :, None]   # 按列归一化

  # with tf.Session() as sess:
    sess = tf.Session()
    guide=sess.run(guide)
    prior_guide=sess.run(prior_guide)

  df_disease_train['guide']=[np.array(i) for i in guide]
  df_disease_train['prior_guide'] = [np.array(i) for i in prior_guide]
  return df_disease_train   # guide是自身存在的特征间关系矩阵（0或1）；prior_guide是特征间条件概率矩阵（0到1）；（batch_size, 1+max_num_codes*2, 1+max_num_codes*2）



def create_matrix_only_cx(df_disease_train, use_prior, use_inf_mask, max_num_codes, prior_scalar):

  batch_size=df_disease_train.shape[0]
  mask=np.array([i for i in df_disease_train['cx_mask']])
  mask=tf.convert_to_tensor(mask,dtype=tf.float32)
  mask = tf.concat([tf.ones(batch_size, dtype=tf.float32)[:, None], mask], axis=1)

  # batch_size = dx_ids.dense_shape[0]
  num_dx_ids = max_num_codes
  num_codes = 1 + num_dx_ids

  guide = None
  if use_inf_mask:
    row0 = tf.concat([tf.ones([1, 1]), tf.ones([1, num_dx_ids])], axis=1)   # (1, 101)  [[0, 1,1,1,1,....,1,0,0,0,0,....,0]]
    row1 = tf.concat([tf.ones([num_dx_ids, 1 + num_dx_ids])],  axis=1)   # (num_dx_ids, 1+num_dx_ids+num_proc_ids)
    # row2 = tf.zeros([num_proc_ids, num_codes])

    guide = tf.concat([row0, row1], axis=0)   # (101,101)
    guide=guide - tf.eye(num_codes)[ :, :]
    # guide = guide + tf.transpose(guide)
    guide = tf.tile(guide[None, :, :], [batch_size, 1, 1])   # (batch_size, 101,101)
    # mask (batch_size, 1+50+50)； mask[:, :, None] (32, 101, 1) ； mask[:, None,:]  (32, 1, 101)； mask[None,:, :]   (1, 32, 101)

    guide = (guide * mask[:, :, None] * mask[:, None, :] + tf.eye(num_codes)[None, :, :])   # (batch_size, 101, 101)
    # guide: 针对每个患者个体，只有在mask中为1的dx和proc对应的行和列, 在上一步生成的guide中才继续保留为1，包含自环（对角均为1）; 每个患者的guide不一样

  prior_guide = None
  if use_prior:
    prior_indices =[i for i in df_disease_train['cx_prior_indices']]
    prior_idx_values = [i for i in df_disease_train['cx_prior_values']]

    prior_idx_dx = []
    for p_i in range(len(prior_indices)):
        prior_idx_dx.extend([[p_i, dx_i[0], dx_i[1]] for dx_i in prior_indices[p_i]])

    prior_idx_values_dx = []
    for i in prior_idx_values:
        prior_idx_values_dx.extend(i)
    # 排序
    prior_idx_dx=pd.DataFrame(prior_idx_dx,columns=['p','d1','d2'])
    prior_idx_dx['v']=prior_idx_values_dx
    prior_idx_dx=prior_idx_dx.sort_values(by=['p','d1','d2'])
    prior_idx_values_dx=prior_idx_dx['v'].values
    prior_idx_dx=prior_idx_dx[['p','d1','d2']].values

    sparse_prior = tf.SparseTensor(indices=prior_idx_dx, values=prior_idx_values_dx, dense_shape=[batch_size, num_dx_ids, num_dx_ids])

    prior_guide = tf.sparse.to_dense(sparse_prior, validate_indices=True)    # dx和proc的条件概率矩阵  （32，100，100）

    prior_guide=tf.cast(prior_guide, tf.float32)
    # 在prior_guide的基础上，添加visit对应行和列
    visit_guide = tf.convert_to_tensor( [prior_scalar] * max_num_codes ,  dtype=tf.float32)    # [0.5 0.5 0.5 0.5 0.5  0.  0.  0.  0.  0. ]
    prior_guide = tf.concat([tf.tile(visit_guide[None, None, :], [batch_size, 1, 1]), prior_guide],axis=1)
    # visit_guide （batch_size, 1, max_num_codes*2）; concat prior_guide后：（batch_size, 1+max_num_codes*2, max_num_codes*2）

    visit_guide = tf.concat([[0.0], visit_guide], axis=0)
    prior_guide = tf.concat([tf.tile(visit_guide[None, :, None], [batch_size, 1, 1]), prior_guide],axis=2)
    # visit_guide  (32, 1+max_num_codes*2, 1); concat prior_guide后：（batch_size, 1+max_num_codes*2, 1+max_num_codes*2）

    # prior_guide = (prior_guide * mask[:, :, None] * mask[:, None, :] +prior_scalar * tf.eye(num_codes)[None, :, :])   # 条件概率矩阵针对患者个体化，并添加自环
    prior_guide = (prior_guide * mask[:, :, None] * mask[:, None, :] +  tf.eye(num_codes)[None, :,  :])  # 条件概率矩阵针对患者个体化，并添加自环

    # degrees = tf.reduce_sum(prior_guide, axis=2)
    # prior_guide = prior_guide / degrees[:, :, None]   # 按列归一化

  # with tf.Session() as sess:
    sess = tf.Session()
    guide=sess.run(guide)
    prior_guide=sess.run(prior_guide)

  df_disease_train['guide_cx']=[np.array(i) for i in guide]
  df_disease_train['prior_guide_cx'] = [np.array(i) for i in prior_guide]
  return df_disease_train   # guide是自身存在的特征间关系矩阵（0或1）；prior_guide是特征间条件概率矩阵（0到1）；（batch_size, 1+max_num_codes*2, 1+max_num_codes*2）



def impute_dx_ints(x, max_num_codes, len_diseases):
    new_x=x.copy()
    imputed=[len_diseases]*(max_num_codes-len(x))
    new_x.extend(imputed)
    return new_x


def get_mask_dx_ints(x, max_num_codes):
    mask=[1]*len(x)
    imputed=[0]*(max_num_codes-len(x))
    mask.extend(imputed)
    return mask



class EarlyStopping:
    def __init__(self, patience=10, min_thred=-0.0001):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.min_thred=min_thred

    def step(self, acc, saver, sess, save_file, max_is_better=True):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(saver, sess, save_file)
        else:
            if max_is_better == True:  # 在评价指标越大越好的情况下
                if score <= self.best_score:
                    self.counter += 1
                    if self.counter % 5 == 0:
                        print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.save_checkpoint(saver, sess, save_file)
                    self.counter = 0
            else:  # 在评价指标越小越好的情况下（如loss）
                # if score >= self.best_score:
                if score - self.best_score >= self. min_thred:
                    self.counter += 1
                    if self.counter % 5 == 0:
                        print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.save_checkpoint(saver, sess, save_file)
                    self.counter = 0
        return self.early_stop, self.best_score

    def save_checkpoint(self, saver, sess, save_file):
        '''Saves model when validation loss decrease.'''
        # torch.save(model.state_dict(), 'es_checkpoint.pt')
        # torch.save(model.state_dict(), output_file + model_name)
        # best_parameters_file = output_file + model_name
        """  """
        saver.save(sess, save_file)



class GCT(object):
    def __init__(self, **params):
        self.weights_init_sd=params['weights_init_sd']
        self.biases_init_value = params['biases_init_value']
        self.dropout = params['dropout']
        self.learning_rate = params['learning_rate']
        self.training_epochs = params['training_epochs']
        self.batch_size = params['batch_size']
        self.display_step = params['display_step']
        self.n_features = params['n_features']
        self.n_hidden_1 = params['n_hidden_1']
        self.n_hidden_2 = params['n_hidden_2']
        self.n_hidden_3 = params['n_hidden_3']
        self.n_classes = params['n_classes']

        self.n_hidden_4 = params['n_hidden_4']

        # params of GCT
        self._vocab_sizes = params['_vocab_sizes']
        self._feature_keys = params['_feature_keys']
        self._embedding_size = params['_embedding_size']
        self._max_num_codes = params['_max_num_codes']   # max_num_codes-dx
        self._use_prior = params['_use_prior']
        self._use_inf_mask = params['_use_inf_mask']
        self._prior_scalar = params['_prior_scalar']
        self._feature_set = params['_feature_set']

        self.num_transformer_stack = params['num_transformer_stack']
        self.num_feedforward = params['num_feedforward']
        self.num_attention_heads = params['num_attention_heads']
        self.ffn_dropout = params['ffn_dropout']

        self.save_path = params['save_path']
        self.patience = params['patience']
        self.early_stop = params['early_stop']

        self.max_num_codes_cx = params['max_num_codes_cx']
        # self.cx_vocab_size = params['cx_vocab_size']

        # self._reg_coef = params['_reg_coef']
        self._reg_coef_cx = params['_reg_coef_cx']
        self._reg_coef_dx = params['_reg_coef_dx']
        self.dic_w = params['dic_w']

    def fully_connected_layer(self, input, weight, bias, keep_prop, activation="relu"):
            layer = tf.add(tf.matmul(input, weight), bias)
            if activation is "tanh":
                layer = tf.nn.tanh(layer)
            elif activation is "sigmoid":
                layer = tf.nn.sigmoid(layer)
            else:
                layer = tf.nn.relu(layer)
            layer = tf.nn.dropout(layer, keep_prob=keep_prop)
            return layer

    def get_loss(self, logits, labels, attentions):
        """Creates a loss tensor.

        Args:
          logits: Logits for prediction. This is obtained by calling get_prediction.
          labels: Labels for prediction.
          attentions: List of attention values from all layers of GCT. This is
            obtained by calling get_prediction.

        Returns:
          Loss tensor. If we use the conditional probability matrix, then GCT's
          attention mechanism will be regularized using KL divergence.
        """

        # attentions： list of attention ;    attention: (batch_size, _num_heads, 1+50+50, 1+50+50)   对dim=3做softmax；

        # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

        # -------------------------
        # 'balanced': ['none', '3year', '5year', '10year'][3]

        # # 计算sample weight
        # if self.balanced!='none':
        #     if self.balanced=='10year':
        #         dic_w = {1: 5.02, 0: 1.25}
        #     if self.balanced == '5year':
        #         dic_w = {1: 8.49, 0: 1.13}
        #     if self.balanced == '3year':
        #         dic_w = {1: 19.43, 0: 1.05}
        sample_weight = tf.cast(labels, tf.float32)[:, 0] * self.dic_w[0] + tf.cast(labels, tf.float32)[:, 1] * self.dic_w[1]
        loss = loss * sample_weight  # 用weight加权
        # -------------------------


        loss = tf.reduce_mean(loss)

        if self._use_prior:
            attention_tensor = tf.convert_to_tensor(attentions)     # shape=(3, ?, 1, 101, 101)

            # attention_tensor = tf.gather(attention_tensor, indices=list(range(1, attention_tensor.shape[-2])), axis=-2)
            # attention_tensor = tf.gather(attention_tensor, indices=list(range(1, attention_tensor.shape[-1])), axis=-1)

            attention_tensor_cx = tf.gather(attention_tensor, indices=list(range(1+self._max_num_codes, attention_tensor.shape[-2])), axis=-2)
            attention_tensor_cx = tf.gather(attention_tensor_cx, indices=list(range(1+self._max_num_codes,attention_tensor.shape[-1])), axis=-1)
            attention_tensor_dx = tf.gather(attention_tensor, indices=list(range(1, 1 + self._max_num_codes)), axis=-2)
            attention_tensor_dx = tf.gather(attention_tensor_dx, indices=list(range(1, 1 + self._max_num_codes)), axis=-1)

            kl_terms_cx = []
            for i in range(1, 1+self.num_transformer_stack):
                log_p = tf.log(attention_tensor_cx[i - 1] + 1e-12)
                log_q = tf.log(attention_tensor_cx[i] + 1e-12)
                kl_term = attention_tensor_cx[i - 1] * (log_p - log_q)   #  shape=(?, 1, 101, 101)
                kl_term = tf.reduce_sum(kl_term, axis=-1)     # shape=(?, 1, 101)
                kl_term = tf.reduce_mean(kl_term)            # shape=()
                kl_terms_cx.append(kl_term)
            reg_term_cx = tf.reduce_mean(kl_terms_cx)

            kl_terms_dx = []
            for i in range(1, 1 + self.num_transformer_stack):
                log_p = tf.log(attention_tensor_dx[i - 1] + 1e-12)
                log_q = tf.log(attention_tensor_dx[i] + 1e-12)
                kl_term = attention_tensor_dx[i - 1] * (log_p - log_q)  # shape=(?, 1, 101, 101)
                kl_term = tf.reduce_sum(kl_term, axis=-1)  # shape=(?, 1, 101)
                kl_term = tf.reduce_mean(kl_term)  # shape=()
                kl_terms_dx.append(kl_term)
            reg_term_dx = tf.reduce_mean(kl_terms_dx)

            loss += self._reg_coef_cx * reg_term_cx+self._reg_coef_dx * reg_term_dx
        return loss, reg_term_dx, reg_term_cx

    def get_init_dis_emd(self, edge_ntw):
        if edge_ntw[0] == 'phi':
            emds = pickle.load(open(dir + '/GCT2023_data/dis_emds/phi_0.001000_final_emd.pkl', "rb"))
            dic_g_no2icd = pickle.load(open(dir + '/GCT2023_data/dis_emds/phi_0.001000_dic_g_no2icd.pkl', "rb"))
        elif edge_ntw[0] == 'CC':
            emds = pickle.load(open(dir + '/GCT2023_data/dis_emds/CC_0.001000_final_emd.pkl', "rb"))
            dic_g_no2icd = pickle.load(open(dir + '/GCT2023_data/dis_emds/CC_0.001000_dic_g_no2icd.pkl', "rb"))
        elif edge_ntw[0] == 'RR':
            emds = pickle.load(open(dir + '/GCT2023_data/dis_emds/RR_0.001000_final_emd_v2.pkl', "rb"))
            dic_g_no2icd = pickle.load(open(dir + '/GCT2023_data/dis_emds/RR_0.001000_dic_g_no2icd_v2.pkl', "rb"))
        elif edge_ntw[0] == 'cc0.001_and_phi0.001':
            emds = pickle.load(open(dir + '/GCT2023_data/dis_emds/PhiCC0.001000_noWeight_final_emd.pkl', "rb"))
            dic_g_no2icd = pickle.load(open(dir + '/GCT2023_data/dis_emds/PhiCC0.001000_noWeight_dic_g_no2icd.pkl', "rb"))
        # emds = pickle.load(open('F:/GCT2023_data/dis_emds/RR_final_emd.pkl', "rb"))
        emds = np.array(emds[120])
        # dic_g_no2icd = pickle.load(open('F:/GCT2023_data/dis_emds/RR_dic_g_no2icd.pkl', "rb"))
        dic_icd2emds = dict(zip(dic_g_no2icd.values(), emds))  # {'B18': np.array([-0.1816,

        dis_dim = emds.shape[-1]

        dis2no = pickle.load(open(dir + '/GCT2023_data/data/dis2no.pkl', "rb"))
        # no2dis=dict(zip(dis2no.values(),dis2no.keys()))

        padding_dis_num = len(dis2no) - len(set(list(dic_icd2emds.keys())) & set(list(dis2no.keys())))
        padding_value = np.random.normal(loc=0.0, scale=0.0, size=(padding_dis_num, dis_dim))

        padding_cur = 0
        init_dis_emd = []
        for i in dis2no.keys():
            if i in dic_icd2emds:
                init_dis_emd.append(dic_icd2emds[i])
            else:
                init_dis_emd.append(padding_value[padding_cur])
                padding_cur += 1

        init_dis_emd = np.array(init_dis_emd)
        return init_dis_emd

    def get_prediction(self,x_train, y_train,x_test, y_test, df_disease_train,df_disease_test,is_training):

        weights = {
            'h1': tf.Variable(tf.truncated_normal(shape=[self.n_features, self.n_hidden_1], stddev=self.weights_init_sd)),
            'h2': tf.Variable(tf.truncated_normal(shape=[self._embedding_size, self.n_hidden_2], stddev=self.weights_init_sd)),
            'h3': tf.Variable(tf.truncated_normal(shape=[self._embedding_size, self.n_hidden_3], stddev=self.weights_init_sd)),
            'h4': tf.Variable(tf.truncated_normal(shape=[self.n_hidden_3, self.n_hidden_4], stddev=self.weights_init_sd)),
            # 'out': tf.Variable(tf.truncated_normal(shape=[self.n_hidden_3+self.n_hidden_4, self.n_classes], stddev=self.weights_init_sd))
            'out': tf.Variable(tf.truncated_normal(shape=[self.n_hidden_4, self.n_classes], stddev=self.weights_init_sd))
        }

        biases = {
            'b1': tf.Variable(tf.constant(self.biases_init_value, shape=[self.n_hidden_1])),
            'b2': tf.Variable(tf.constant(self.biases_init_value, shape=[self.n_hidden_2])),
            'b3': tf.Variable(tf.constant(self.biases_init_value, shape=[self.n_hidden_3])),
            'b4': tf.Variable(tf.constant(self.biases_init_value, shape=[self.n_hidden_4])),
            'b_out': tf.Variable(tf.constant(self.biases_init_value, shape=[self.n_classes]))
        }

        x_ = tf.placeholder(tf.float32, [None, self.n_features])
        y_ = tf.placeholder(tf.int32, [None, self.n_classes])
        keep_prob_ = tf.placeholder(tf.float32)
        lr_ = tf.placeholder(tf.float32)

        feat_dx_ints_ = tf.placeholder(tf.int32, [None, self._max_num_codes])
        mask_dx_ints_ = tf.placeholder(tf.int32, [None, self._max_num_codes])
        # for bucket clinical features
        feat_cx_ints_ = tf.placeholder(tf.int32, [None, self.max_num_codes_cx])
        mask_cx_ints_ = tf.placeholder(tf.int32, [None, self.max_num_codes_cx])

        if self._feature_set == 'only_dx':
            guide_= tf.placeholder(tf.float32, [None, self._max_num_codes+1, self._max_num_codes+1])
            prior_guide_= tf.placeholder(tf.float32, [None, self._max_num_codes+1, self._max_num_codes+1])
        #     guide, prior_guide = create_matrix_only_dx(features, masks, self._use_prior,  self._use_inf_mask, self._max_num_codes,  self._prior_scalar)
        elif self._feature_set=='dx_cx':    # todo 2. 设置这个超参数
            guide_ = tf.placeholder(tf.float32, [None, self._max_num_codes+self.max_num_codes_cx + 1, self._max_num_codes+self.max_num_codes_cx + 1])
            prior_guide_ = tf.placeholder(tf.float32, [None, self._max_num_codes+self.max_num_codes_cx + 1, self._max_num_codes+self.max_num_codes_cx + 1])
        else: sys.exit(0)
        print('self._vocab_sizes, self._feature_keys',self._vocab_sizes,self._feature_keys)

        # init_dis_emd=self.get_init_dis_emd(edge_ntw)    # 获得初始化的disease embedding
        init_dis_emd=np.array([])

        cx_featrue_embedder=FeatureEmbedder(self._vocab_sizes,self._feature_keys, self._embedding_size, init_dis_emd)
        cx_embedding_dict, cx_mask_dict =cx_featrue_embedder.lookup({'dx_ints':feat_dx_ints_,    'cx_ints':feat_cx_ints_},
                                                                                                             {'dx_ints':mask_dx_ints_, 'cx_ints':mask_cx_ints_})   # feat_cx_ints_补齐的feat_cx_idx

        keys = ['visit'] +self._feature_keys    # feature_keys=['dx_ints', 'proc_ints']
        cx_embeddings = tf.concat([cx_embedding_dict[key] for key in keys], axis=1)  # (batch_size, 1+50+50, embedding_size);
        cx_masks = tf.concat([cx_mask_dict[key] for key in keys], axis=1)  # masks (batch_size, 1+50+50)
        # print('embeddings,masks',embeddings,masks)   # Tensor("concat_1:0", shape=(?, 31, 128), dtype=float32) Tensor("concat_2:0", shape=(?, 31), dtype=int32)
        cx_masks = tf.cast(cx_masks, tf.float32)

        # print('cx_embeddings', cx_embeddings)
        # print('cx_masks', cx_masks)

        modelGCT = GraphConvolutionalTransformer(embedding_size=self._embedding_size,
                                                 num_transformer_stack=self.num_transformer_stack,
                                                 num_feedforward=self.num_feedforward,
                                                 num_attention_heads=self.num_attention_heads,
                                                 ffn_dropout=self.ffn_dropout,
                                                 attention_normalizer='softmax',
                                                 multihead_attention_aggregation='concat',
                                                 directed_attention=False,
                                                 use_inf_mask=self._use_inf_mask,
                                                 use_prior=self._use_prior    # 用来处理临床特征
                                                 )
        cx_hidden, cx_attentions = modelGCT.get_emb_atte(cx_embeddings, cx_masks[:, :, None], guide_, prior_guide_, keep_prob_)
        # print('cx_hidden',cx_hidden)
        # print('cx_attentions',cx_attentions)
        cx_pre_logit = cx_hidden[:, 0, :]  # 取visit的编码   pre_logit :  (batch, embedding_size)
        cx_pre_logit = tf.reshape(cx_pre_logit, [-1, self._embedding_size])

        # layer_2 = self.fully_connected_layer(cx_pre_logit, weights['h2'], biases['b2'], keep_prob_)
        layer_3 = self.fully_connected_layer(cx_pre_logit, weights['h3'], biases['b3'], keep_prob_)
        layer_4 = self.fully_connected_layer(layer_3, weights['h4'], biases['b4'], keep_prob_)
        out = tf.add(tf.matmul(layer_4, weights['out']), biases['b_out'])



        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y_))  # todo 交叉熵损失函数
        # # cost=focal_loss(out,  tf.cast(y_, dtype=tf.float32), alpha=0.5, gamma=0)
        cost, reg_term_dx, reg_term_cx = self.get_loss(logits=out, labels=y_, attentions=cx_attentions)

        # print('cost',cost)  #是一个数字
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_).minimize(cost)
        y_score = tf.nn.softmax(logits=out)


        # 处理feed_dict中的特征
        feat_dx_ints_train = np.array([i for i in df_disease_train['ls_dx_ints_imputed']])
        mask_dx_ints_train = np.array([i for i in df_disease_train['mask']])
        feat_dx_ints_test = np.array([i for i in df_disease_test['ls_dx_ints_imputed']])
        mask_dx_ints_test = np.array([i for i in df_disease_test['mask']])

        # guide_dx_cx,prior_guide_dx_cx
        guide_train = np.array([i for i in df_disease_train['guide_dx_cx']])
        prior_guide_train = np.array([i for i in df_disease_train['prior_guide_dx_cx']])
        guide_test = np.array([i for i in df_disease_test['guide_dx_cx']])
        prior_guide_test = np.array([i for i in df_disease_test['prior_guide_dx_cx']])


        # bucket feature特征
        feat_cx_ints_train = np.array([i for i in df_disease_train['ls_cx_ints_imputed']])
        mask_cx_ints_train = np.array([i for i in df_disease_train['cx_mask']])
        feat_cx_ints_test = np.array([i for i in df_disease_test['ls_cx_ints_imputed']])
        mask_cx_ints_test = np.array([i for i in df_disease_test['cx_mask']])


        ## initiate training logs
        loss_rec = np.zeros([self.training_epochs, 1])
        training_eval = np.zeros([self.training_epochs, 2])
        testing_eval = np.zeros([self.training_epochs, 2])

        # early stop
        early_stop = self.early_stop
        stopper = EarlyStopping(patience=self.patience, min_thred=-0.0001)  # 早停法
        saver = tf.train.Saver()

        if is_training:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                total_batch = int(np.shape(x_train)[0] / self.batch_size)

                ## Training cycle
                t = time.time()
                for epoch in range(self.training_epochs):
                    t0 = time.time()
                    avg_cost = 0.
                    x_tmp, y_tmp, feat_dx_ints_tmp, mask_dx_ints_tmp, guide_tmp, prior_guide_tmp,  feat_cx_ints_tmp, mask_cx_ints_tmp= \
                        shuffle(x_train, y_train, feat_dx_ints_train, mask_dx_ints_train, guide_train, prior_guide_train, feat_cx_ints_train, mask_cx_ints_train)
                    # Loop over all batches
                    for i in range(total_batch - 1):
                        batch_x, batch_y = x_tmp[i * self.batch_size:i * self.batch_size + self.batch_size],   y_tmp[i * self.batch_size:i * self.batch_size + self.batch_size]
                        batch_feat_dx_ints, batch_mask_dx_ints = feat_dx_ints_tmp[i * self.batch_size:i * self.batch_size + self.batch_size], mask_dx_ints_tmp[i * self.batch_size:i * self.batch_size + self.batch_size]
                        batch_guide, batch_prior_guide = guide_tmp[ i * self.batch_size:i * self.batch_size + self.batch_size], prior_guide_tmp[i * self.batch_size:i * self.batch_size + self.batch_size]
                        batch_feat_cx_ints, batch_mask_cx_ints = feat_cx_ints_tmp[i * self.batch_size:i * self.batch_size + self.batch_size], mask_cx_ints_tmp[i * self.batch_size:i * self.batch_size + self.batch_size]


                        _, c = sess.run([optimizer, cost],
                                        feed_dict={x_: batch_x, y_: batch_y, keep_prob_: 1 - self.dropout, lr_: self.learning_rate,
                                                   feat_dx_ints_: batch_feat_dx_ints, mask_dx_ints_: batch_mask_dx_ints,  guide_: batch_guide, prior_guide_: batch_prior_guide,
                                                   feat_cx_ints_: batch_feat_cx_ints, mask_cx_ints_: batch_mask_cx_ints})

                        # Compute average loss
                        avg_cost += c / total_batch

                    del x_tmp, y_tmp

                    ## Display logs per epoch step
                    if epoch % self.display_step == 0:
                        ## Monitor training
                        loss_rec[epoch] = avg_cost
                        y_s, tr_reg_term_dx, tr_reg_term_cx, tr_cx_attentions = sess.run([y_score, reg_term_dx, reg_term_cx, cx_attentions], feed_dict={x_: x_train, y_: y_train, keep_prob_: 1,
                                                             feat_dx_ints_: feat_dx_ints_train, mask_dx_ints_: mask_dx_ints_train,
                                                             guide_: guide_train, prior_guide_: prior_guide_train,
                                                             feat_cx_ints_: feat_cx_ints_train, mask_cx_ints_: mask_cx_ints_train})

                        y_s = np.reshape(np.array(y_s), [np.shape(x_train)[0], 2])[:, 1]
                        acc = metrics.accuracy_score(y_train[:, 1], y_s > 0.5)
                        auc = metrics.roc_auc_score(y_train[:, 1], y_s)
                        # precision = metrics.precision_score(y_train[:, 1], y_s > 0.5)
                        # recall = metrics.recall_score(y_train[:, 1], y_s > 0.5)
                        f1 = metrics.f1_score(y_train[:, 1], y_s > 0.5)
                        # _precision, _recall, _threshold = metrics.precision_recall_curve(y_train[:, 1], y_s)
                        # PR = metrics.auc(_recall, _precision)
                        training_eval[epoch] = [acc, auc]

                        dur = (time.time() - t0) / 60

                        ## Testing
                        y_s, val_cost = sess.run([y_score, cost], feed_dict={x_: x_test, y_: y_test, keep_prob_: 1,
                                                                             feat_dx_ints_: feat_dx_ints_test,  mask_dx_ints_: mask_dx_ints_test,
                                                                             guide_: guide_test, prior_guide_: prior_guide_test,
                                                                             feat_cx_ints_: feat_cx_ints_test,  mask_cx_ints_: mask_cx_ints_test})

                        y_s = np.reshape(np.array(y_s), [np.shape(x_test)[0], 2])[:, 1]
                        test_acc = metrics.accuracy_score(y_test[:, 1], y_s > 0.5)
                        test_auc = metrics.roc_auc_score(y_test[:, 1], y_s)
                        # test_prec = metrics.precision_score(y_test[:, 1], y_s > 0.5)
                        # test_recall = metrics.recall_score(y_test[:, 1], y_s > 0.5)
                        test_f1 = metrics.f1_score(y_test[:, 1], y_s > 0.5)
                        # _precision, _recall, _threshold = metrics.precision_recall_curve(y_test[:, 1], y_s)
                        # test_pr = metrics.auc(_recall, _precision)
                        print("Epc {:03d}, TrnLoss {:.4f} , ValLoss {:.4f} | TrnAuc {:.4f}, Acc {:.4f}, F1 {:.4f} | ValAuc {:.4f}, Acc {:.4f}, F1 {:.4f} | "
                              "T {:.4f}  || reg_term_dx {:.4f} ,  reg_term_cx {:.4f} , ".format(epoch, avg_cost, val_cost, round(auc, 4),  round(acc, 4), round(f1, 4),
                                                                                                 round(test_auc, 4),  round(test_acc, 4), round(test_f1, 4),np.mean(dur),
                                                                                                 round(tr_reg_term_dx, 4),  round(tr_reg_term_cx, 4)))

                        # 早停
                        if early_stop == True and epoch>=19:
                            is_stop, best_score = stopper.step(val_cost, saver, sess, self.save_path, max_is_better=False)
                            if is_stop == True:
                                break
                # pickle.dump(tr_cx_attentions, open(dir+'/GCT2023_data/tmp/tr_cx_attentions_v2.pkl', "wb"))
                if not early_stop:
                    saver.save(sess, self.save_path)
        else:  # 测试
            saver = tf.train.Saver()
            with tf.Session() as sess:
                # sess.run(init_op)  # 可以执行或不执行，restore的值会override初始值
                saver.restore(sess, self.save_path)

                ## Testing
                y_s, final_cx_hidden = sess.run([y_score,cx_hidden], feed_dict={x_: x_test, y_: y_test, keep_prob_: 1,
                                                     feat_dx_ints_: feat_dx_ints_test,mask_dx_ints_: mask_dx_ints_test,
                                                     guide_: guide_test, prior_guide_: prior_guide_test,
                                                     feat_cx_ints_: feat_cx_ints_test,mask_cx_ints_: mask_cx_ints_test})

                y_s = np.reshape(np.array(y_s), [np.shape(x_test)[0], 2])[:, 1]
                test_acc = metrics.accuracy_score(y_test[:, 1], y_s > 0.5)
                test_auc = metrics.roc_auc_score(y_test[:, 1], y_s)
                test_prec = metrics.precision_score(y_test[:, 1], y_s > 0.5)
                test_recall = metrics.recall_score(y_test[:, 1], y_s > 0.5)
                test_f1 = metrics.f1_score(y_test[:, 1], y_s > 0.5)
                _precision, _recall, _threshold = metrics.precision_recall_curve(y_test[:, 1], y_s)
                test_pr = metrics.auc(_recall, _precision)


                # pickle.dump(final_dx_hidden, open('F:/GCT2023_data/tmp/final_dx_hidden_woCI_seed0_cv1__.pkl',"wb"))

                print('test_auc: %.3f, test_pr: %.3f, test_f1: %.3f, test_recall: %.3f, test_prec: %.3f, test_acc: %.3f' % (
                    test_auc, test_pr, test_f1, test_recall, test_prec, test_acc))
                # print("*****=====", "Testing accuracy: ", round(acc, 3), " Testing auc: ", round(auc, 3), "=====*****")
                return test_auc, test_pr, test_f1, test_recall, test_prec, test_acc


def get_cols(x,ls_ntw_dis):
    ls=[]
    for i in range(len(x)):
        if x[i] in ls_ntw_dis:
            ls.append(i)
    return np.array(ls)



def update_prior_guide(prior_guide, prior_guide_ntw, ntw_dis_idx):
    # print('ntw_dis_idx',ntw_dis_idx)
    if ntw_dis_idx.shape[0]>0:
        # prior_guide[ntw_dis_idx,ntw_dis_idx]=prior_guide_ntw[ntw_dis_idx,ntw_dis_idx]
        ntw_dis_idx+=1
        prior_guide[ntw_dis_idx, :] = prior_guide_ntw[ntw_dis_idx, :]
        prior_guide[:, ntw_dis_idx] = prior_guide_ntw[:, ntw_dis_idx]
    # print('prior_guide',prior_guide)
    return prior_guide


def ps_nomalization(x):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    degrees = tf.reduce_sum(x, axis=-1)
    # print('degrees',x, degrees,degrees[:, None] )
    x = x / degrees[:, None]   # 按列归一化
    sess=tf.Session()
    x=sess.run(x)
    return x



def main(rpt,edge,edge_ntw,tmp_reg_coef_dx,tmp_reg_coef_cx, dir, col_year, dic_w):

    if col_year=='label_3year':
        year_=3
    elif col_year == 'label_5year':
        year_ = 5

    for has_CIs in [False]:
        ci_flag = 'wtCI' if has_CIs else 'woCI'

        ls_rsts_=[]

        for seed in range(10):

            print('===== NN    round    %s    seed%d    ======='%(ci_flag,seed))

            trainIDs_ls = pickle.load( open(dir + '/GCT2023_data/data/trainIDs_ls_seed%d_%s_year%d.pkl' % (seed, ci_flag, year_), "rb"))
            valIDs_ls = pickle.load( open(dir + '/GCT2023_data/data/valIDs_ls_seed%d_%s_year%d.pkl' % (seed, ci_flag, year_), "rb"))
            testIDs_ls = pickle.load( open(dir + '/GCT2023_data/data/testIDs_ls_seed%d_%s_year%d.pkl' % (seed, ci_flag, year_), "rb"))

            dic_train = pickle.load(open(dir + '/GCT2023_data/data/dic_train_seed%d_%s_condiP_%s_%s_year%d.pkl' % (seed, ci_flag, edge[0], edge[1], year_), 'rb'))
            dic_test = pickle.load(open(dir + '/GCT2023_data/data/dic_test_seed%d_%s_condiP_%s_%s_year%d.pkl' % (seed, ci_flag, edge[0], edge[1], year_), 'rb'))

            # # 选择标签（年份）
            # # col_year='label_10year'
            # pID_labels=pickle.load(open(dir + '/GCT2023_data/data/pID_labels.pkl', 'rb'))    # ['patient_id','label','label_5year','label_10year','label_3year']
            # for j in list(dic_train.keys()):
            #     new_y=pd.merge(pd.DataFrame(dic_train[j][0],columns=['patient_id']),pID_labels, on='patient_id')
            #     new_y=new_y[col_year].values
            #     dic_train[j][2]=new_y
            # for j in list(dic_test.keys()):
            #     new_y=pd.merge(pd.DataFrame(dic_test[j][0],columns=['patient_id']),pID_labels, on='patient_id')
            #     new_y=new_y[col_year].values
            #     dic_test[j][2]=new_y


            print(len(dic_train[0]))  # 3
            print(dic_train[0][1].shape)
            #   dic_train       :  [train_ids: np.array, basic_feats: ndarray, train_y: np.array, df_disease_p]
            #   df_disease_p:   ['patient_id', 'ls_dx_ints', 'ls_dx_names', 'dx_prior_indices', 'dx_prior_values']

            #===================
            # dic_train_ntw=pickle.load(open(dir+'/GCT2023_data/data/dic_train_seed%d_%s_DCN_%s_%s.pkl'%(seed,ci_flag,'CC','0.001000'), 'rb'))
            # dic_test_ntw = pickle.load(open(dir + '/GCT2023_data/data/dic_test_seed%d_%s_DCN_%s_%s.pkl' % (seed, ci_flag, 'CC', '0.001000'), 'rb'))
            # dic_train_ntw = pickle.load( open(dir + '/GCT2023_data/data/dic_train_seed%d_%s_DCN_%s_%s.pkl' % (seed, ci_flag, edge_ntw[0], edge_ntw[1] ),  'rb'))
            # dic_test_ntw = pickle.load( open(dir + '/GCT2023_data/data/dic_test_seed%d_%s_DCN_%s_%s.pkl' % (seed, ci_flag, edge_ntw[0],edge_ntw[1] ), 'rb'))
            dic_train_ntw = pickle.load(open(dir + '/GCT2023_data/data/dic_train_seed%d_%s_DCN_%s_%s_year%d.pkl' % ( seed, ci_flag, edge_ntw[0], edge_ntw[1], year_), 'rb'))
            dic_test_ntw = pickle.load(open(dir + '/GCT2023_data/data/dic_test_seed%d_%s_DCN_%s_%s_year%d.pkl' % (seed, ci_flag, edge_ntw[0], edge_ntw[1], year_), 'rb'))
            #====================

            ls_rst=[]
            for cv_i in range(10):

                train_val_samples=dic_train[cv_i][1]

                train_index = np.zeros_like(trainIDs_ls[cv_i])   # 按照trainIDs的顺序来排的
                for i, aa in np.ndenumerate(trainIDs_ls[cv_i]):
                    train_index[i] = np.where(dic_train[cv_i][0] == aa)[0]

                val_index = np.zeros_like(valIDs_ls[cv_i])
                for i, aa in np.ndenumerate(valIDs_ls[cv_i]):
                    val_index[i] = np.where(dic_train[cv_i][0] == aa)[0]

                if set(list(train_index))&set(list(val_index)) or len(set(list(train_index))|set(list(val_index)))!=train_val_samples.shape[0]:
                    raise Exception('数据划分有问题')

                # expression_train=train_val_samples.loc[train_index].values
                expression_train = train_val_samples[train_index]
                y_train=dic_train[cv_i][2][train_index]
                # expression_val = train_val_samples.loc[val_index].values
                expression_val = train_val_samples[val_index]
                y_val = dic_train[cv_i][2][val_index]
                # expression_test=dic_test[cv_i][1].values
                expression_test = dic_test[cv_i][1]
                y_test=dic_test[cv_i][2]


                # clinical features
                x_train = expression_train[:, :]
                x_val = expression_val[:, :]
                x_test = expression_test[:, :]


                # 分桶后的clinical features
                # ['patient_id', 'cx_bucket', 'cx_bucket_idx', 'ls_cx_ints_imputed', 'cx_mask', 'cx_prior_indices', 'cx_prior_values',
                #   cx_oc, dx_oc, cx_oc_imputed, dx_oc_imputed]
                bucket_cx=pickle.load(open(dir+'/GCT2023_data/data_xgb_GCT/%s_pID_cxBucket_cxIdx___%s_seed%d_cv%d_v2.pkl' % (col_year, ci_flag, seed, cv_i), 'rb'))
                bucket_cx_train = pd.merge(pd.DataFrame(trainIDs_ls[cv_i], columns=['patient_id']), bucket_cx, on='patient_id')
                bucket_cx_val = pd.merge(pd.DataFrame(valIDs_ls[cv_i], columns=['patient_id']), bucket_cx, on='patient_id')
                bucket_cx_test = pd.merge(pd.DataFrame(testIDs_ls[cv_i], columns=['patient_id']), bucket_cx, on='patient_id')

                max_num_codes_cx=pickle.load(open(dir+'/GCT2023_data/data_xgb_GCT/%s_max_cxBucket_num___%s_seed%d_cv%d_v2.pkl' % (col_year, ci_flag, seed, cv_i), 'rb'))
                dic_cxBucket2Idx = pickle.load(open(dir+'/GCT2023_data/data_xgb_GCT/%s_dic_cxBucket2Idx___%s_seed%d_cv%d_v2.pkl' % (col_year, ci_flag, seed, cv_i), 'rb'))
                cx_vocab_size=len(dic_cxBucket2Idx)


                # one-hot encode the labels
                y_train = np.stack((1-y_train, y_train), axis=1)
                y_val = np.stack((1 - y_val, y_val), axis=1)
                y_test = np.stack((1 - y_test, y_test), axis=1)

                # 网络参数矩阵
                #   df_disease_p:   ['patient_id', 'ls_dx_ints', 'ls_dx_names', 'dx_prior_indices', 'dx_prior_values']
                df_disease_train = dic_train[cv_i][-1].iloc[train_index]
                df_disease_val = dic_train[cv_i][-1].iloc[val_index]
                df_disease_test = dic_test[cv_i][-1]

                # 补全dx_features 和 构建 mask矩阵
                dis2no = pickle.load(open(dir+'/GCT2023_data/data/dis2no_seed%d_%s_DCN_%s_%s_year%d.pkl'%(seed,ci_flag,edge_ntw[0],edge_ntw[1], year_), "rb"))
                len_diseases = len(dis2no)
                max_num_codes_dx = 30


                #=====================
                df_disease_ntw_train = dic_train_ntw[cv_i][-1].iloc[train_index]
                df_disease_ntw_val = dic_train_ntw[cv_i][-1].iloc[val_index]
                df_disease_ntw_test = dic_test_ntw[cv_i][-1]
                df_disease_ntw_train['mask'] = df_disease_ntw_train['ls_dx_ints'].apply( lambda x: get_mask_dx_ints(x, max_num_codes_dx))
                df_disease_ntw_val['mask'] = df_disease_ntw_val['ls_dx_ints'].apply( lambda x: get_mask_dx_ints(x, max_num_codes_dx))
                df_disease_ntw_test['mask'] = df_disease_ntw_test['ls_dx_ints'].apply( lambda x: get_mask_dx_ints(x, max_num_codes_dx))
                #=====================


                df_disease_train['ls_dx_ints_imputed'] = df_disease_train['ls_dx_ints'].apply( lambda x: impute_dx_ints(x, max_num_codes_dx, len_diseases))
                df_disease_train['mask'] = df_disease_train['ls_dx_ints'].apply( lambda x: get_mask_dx_ints(x, max_num_codes_dx))
                df_disease_val['ls_dx_ints_imputed'] = df_disease_val['ls_dx_ints'].apply( lambda x: impute_dx_ints(x, max_num_codes_dx, len_diseases))
                df_disease_val['mask'] = df_disease_val['ls_dx_ints'].apply(lambda x: get_mask_dx_ints(x, max_num_codes_dx))
                df_disease_test['ls_dx_ints_imputed'] = df_disease_test['ls_dx_ints'].apply( lambda x: impute_dx_ints(x, max_num_codes_dx, len_diseases))
                df_disease_test['mask'] = df_disease_test['ls_dx_ints'].apply( lambda x: get_mask_dx_ints(x, max_num_codes_dx))
                #   df_disease_p:   ['patient_id', 'ls_dx_ints', 'ls_dx_names', 'dx_prior_indices', 'dx_prior_values','ls_dx_ints_imputed','mask']

                df_disease_train=pd.merge(df_disease_train, bucket_cx_train, on='patient_id')
                df_disease_val = pd.merge(df_disease_val, bucket_cx_val, on='patient_id')
                df_disease_test = pd.merge(df_disease_test, bucket_cx_test, on='patient_id')
                #   df_disease_p:   ['patient_id', 'ls_dx_ints', 'ls_dx_names', 'dx_prior_indices', 'dx_prior_values','ls_dx_ints_imputed','mask',
                # , 'cx_bucket', 'cx_bucket_idx', 'ls_cx_ints_imputed', 'cx_mask', 'cx_prior_indices', 'cx_prior_values',
                #   cx_oc, dx_oc, cx_oc_imputed, dx_oc_imputed]

                params = {'weights_init_sd': 0.1,
                          'biases_init_value': 0.,
                          'dropout': 0.3,  # !
                          'learning_rate': 0.0001,  # !
                          'training_epochs': 50,  # !
                          'batch_size': 8,  # !
                          'display_step': 1,
                          'n_features': np.shape(expression_train)[-1],
                          'n_hidden_1': np.shape(expression_train)[-1],
                          'n_hidden_2': 32,
                          'n_hidden_3': 32,
                          'n_hidden_4': 32,  # !
                          'n_classes': 2,

                          # GCT
                          '_vocab_sizes': {'dx_ints': len_diseases, 'cx_ints': cx_vocab_size},
                          '_feature_keys': ['dx_ints', 'cx_ints'],
                          '_embedding_size': 32,  # !
                          '_max_num_codes': max_num_codes_dx,
                          '_use_prior': True,
                          '_use_inf_mask': True,
                          '_prior_scalar': 0.3,  # !
                          '_feature_set': 'dx_cx',
                          'num_transformer_stack': 1,  # !
                          'num_feedforward': 3,  # !
                          'num_attention_heads': 1,  # !
                          'ffn_dropout': 0.01,

                          'save_path':dir+"/software_test_models/model_%s_seed%d_cv%d.ckpt" % (ci_flag, seed, cv_i),
                          'patience': 5,
                          'early_stop': True,

                          # for bucket clinical features
                          'max_num_codes_cx': max_num_codes_cx,
                          # 'cx_vocab_size': {'dx_ints': cx_vocab_size},
                          #
                          # '_reg_coef': 0.3
                          '_reg_coef_cx': tmp_reg_coef_cx,
                          '_reg_coef_dx': tmp_reg_coef_dx,    # 0.0015,
                          'dic_w': dic_w,     # ['none','3year','5year', '10year'][3],
                          }


                #   df_disease_p:   ['patient_id', 'ls_dx_ints', 'ls_dx_names', 'dx_prior_indices', 'dx_prior_values','ls_dx_ints_imputed','mask', 'guide','prior_guide']
                # 构建guide矩阵和prior_guide矩阵
                df_disease_train = create_matrix_only_dx(df_disease_train, use_prior=True, use_inf_mask=True,  max_num_codes=max_num_codes_dx, prior_scalar=params['_prior_scalar'])
                df_disease_val = create_matrix_only_dx(df_disease_val, use_prior=True, use_inf_mask=True, max_num_codes=max_num_codes_dx, prior_scalar=params['_prior_scalar'])
                df_disease_test = create_matrix_only_dx(df_disease_test, use_prior=True, use_inf_mask=True,  max_num_codes=max_num_codes_dx, prior_scalar=params['_prior_scalar'])

                #==========================
                df_disease_ntw_train = create_ps_matrix_only_dx(df_disease_ntw_train, use_prior=True, use_inf_mask=True, max_num_codes=max_num_codes_dx,  prior_scalar=params['_prior_scalar'])
                df_disease_ntw_val = create_ps_matrix_only_dx(df_disease_ntw_val, use_prior=True, use_inf_mask=True, max_num_codes=max_num_codes_dx,  prior_scalar=params['_prior_scalar'])
                df_disease_ntw_test = create_ps_matrix_only_dx(df_disease_ntw_test, use_prior=True, use_inf_mask=True, max_num_codes=max_num_codes_dx,  prior_scalar=params['_prior_scalar'])
                # ‘prior_guide_ntw’

                df_disease_ntw_train['ntw_dis_idx']=df_disease_ntw_train['ls_dx_names'].apply(lambda x: get_cols(x,list(dis2no.keys())))
                df_disease_ntw_val['ntw_dis_idx'] = df_disease_ntw_val['ls_dx_names'].apply(  lambda x: get_cols(x, list(dis2no.keys())))
                df_disease_ntw_test['ntw_dis_idx'] = df_disease_ntw_test['ls_dx_names'].apply( lambda x: get_cols(x, list(dis2no.keys())))

                df_disease_train=pd.merge(df_disease_train,df_disease_ntw_train[['patient_id','prior_guide_ntw','ntw_dis_idx']],on='patient_id')
                df_disease_val = pd.merge(df_disease_val, df_disease_ntw_val[['patient_id', 'prior_guide_ntw', 'ntw_dis_idx']],  on='patient_id')
                df_disease_test = pd.merge(df_disease_test, df_disease_ntw_test[['patient_id', 'prior_guide_ntw', 'ntw_dis_idx']],  on='patient_id')
                # print('now1')
                df_disease_train['prior_guide']=df_disease_train.apply(lambda row: update_prior_guide(row['prior_guide'],row['prior_guide_ntw'], row['ntw_dis_idx']), axis=1)
                df_disease_val['prior_guide'] = df_disease_val.apply(lambda row: update_prior_guide(row['prior_guide'], row['prior_guide_ntw'], row['ntw_dis_idx']), axis=1)
                df_disease_test['prior_guide'] = df_disease_test.apply(lambda row: update_prior_guide(row['prior_guide'], row['prior_guide_ntw'], row['ntw_dis_idx']), axis=1)
                # print('now2')
                # #==========================


                # 构建guide_cx矩阵和prior_guide_cx矩阵
                df_disease_train = create_matrix_only_cx(df_disease_train, use_prior=True, use_inf_mask=True, max_num_codes=max_num_codes_cx, prior_scalar=params['_prior_scalar'])
                df_disease_val = create_matrix_only_cx(df_disease_val, use_prior=True, use_inf_mask=True, max_num_codes=max_num_codes_cx, prior_scalar=params['_prior_scalar'])
                df_disease_test = create_matrix_only_cx(df_disease_test, use_prior=True, use_inf_mask=True, max_num_codes=max_num_codes_cx, prior_scalar=params['_prior_scalar'])
                # 构建guide_dx_cx矩阵和prior_guide_dx_cx矩阵
                df_disease_train = create_matrix_dx_cx(df_disease_train)   # todo     1.  check
                df_disease_val = create_matrix_dx_cx(df_disease_val)
                df_disease_test = create_matrix_dx_cx(df_disease_test)


                #============================
                train_pg = tf.convert_to_tensor(np.array([_ for _ in df_disease_train['prior_guide_dx_cx']]), dtype=tf.float32)
                degrees = tf.reduce_sum(train_pg, axis=2)
                train_pg = train_pg / degrees[:, :, None]  # 按列归一化

                val_pg = tf.convert_to_tensor(np.array([_ for _ in df_disease_val['prior_guide_dx_cx']]), dtype=tf.float32)
                degrees = tf.reduce_sum(val_pg, axis=2)
                val_pg = val_pg / degrees[:, :, None]  # 按列归一化

                test_pg = tf.convert_to_tensor(np.array([_ for _ in df_disease_test['prior_guide_dx_cx']]), dtype=tf.float32)
                degrees = tf.reduce_sum(test_pg, axis=2)
                test_pg = test_pg / degrees[:, :, None]  # 按列归一化

                print(train_pg, val_pg, test_pg)
                sess = tf.Session()
                df_disease_train['prior_guide_dx_cx'] = [_ for _ in sess.run(train_pg)]
                df_disease_val['prior_guide_dx_cx'] = [_ for _ in sess.run(val_pg)]
                df_disease_test['prior_guide_dx_cx'] = [_ for _ in sess.run(test_pg)]
                # print("df_disease_test['prior_guide_dx_cx'].iloc[0]", df_disease_test['prior_guide_dx_cx'].iloc[0])
                # pickle.dump(df_disease_train['prior_guide_dx_cx'],open(dir+'/GCT2023_data/tmp/prior_guide2.pkl','wb'))
                # print('[[[[[[[[[[[[[[[[[[[[[[[[[[[[[')
                #============================

                #   df_disease_p:
                #   ['patient_id',
                #   'ls_dx_names', 'ls_dx_ints', 'dx_prior_indices', 'dx_prior_values','ls_dx_ints_imputed','mask',guide','prior_guide',
                #  'cx_bucket', 'cx_bucket_idx', 'cx_prior_indices', 'cx_prior_values', 'ls_cx_ints_imputed', 'cx_mask',  'guide_cx',  'prior_guide_cx',
                #  'guide_dx_cx',  'prior_guide_dx_cx']


                model=GCT(**params)

                tf.reset_default_graph()
                model.get_prediction(x_train, y_train,x_val, y_val,df_disease_train,df_disease_val, is_training=True)

                tf.reset_default_graph()
                # pickle.dump(df_disease_test['ls_dx_names'], open('F:/GCT2023_data/tmp/ls_dx_names_seed0_cv1__.pkl', "wb"))
                test_auc, test_pr, test_f1, test_recall, test_prec, test_acc=model.get_prediction(x_train, y_train,x_test, y_test,df_disease_train,df_disease_test, is_training=False)
                ls_rst.append([test_auc, test_pr, test_f1, test_recall, test_prec, test_acc])

            for i in ls_rst:
                print(i)

            ls_rst=np.array(ls_rst)
            m = ls_rst.mean(axis=0)
            s = ls_rst.std(axis=0)
            ls_rst=list(ls_rst)
            ls_rst.append(m)
            ls_rst.append(s)

            print('tmp_reg_coef_dx,tmp_reg_coef_cx',tmp_reg_coef_dx,tmp_reg_coef_cx)
            print('均值', m)
            print('方差', s)

            # pd.DataFrame(ls_rst, columns=['AUC', 'PR', 'F1_score', 'Recall', 'Precision', 'Acc']).to_excel('F:/GCT2023_data/baselines/forgeNet_NN.xlsx',index=False)
            rst_tmp=pd.DataFrame(ls_rst, columns=['AUC', 'PR', 'F1_score', 'Recall', 'Precision', 'Acc'])
            rst_tmp['seed']=seed
            rst_tmp['has_CI']=ci_flag
            ls_rsts_.append(rst_tmp)

            # pd.concat(ls_rsts_,axis=0).to_excel(dir+'/GCT2023_data/baselines/tmp_ours_3year/core_ours_%s_rpt%d_DCN_%s_dicW1-%f.xlsx'%(ci_flag,rpt,col_year,dic_w[1]))
            pd.concat(ls_rsts_,axis=0).to_excel(dir+'/software_test_rsts/software_test_%s_rpt%d_DCN_%s_dicW1-%f.xlsx'%(ci_flag,rpt,col_year,dic_w[1]))



def main_(dir, col_year, dic_w, rpt):
    # tensorflow==1.15.0
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # col_year = 'label_10year'
    # dir = 'E:/0_YangPing/GCT_2023'
    # dir='F:'

    # for rpt in range(3,5):
    for tmp_reg_coef in [(0.015, 0.015)]:    # (0.01, 0.005)!
        tmp_reg_coef_dx, tmp_reg_coef_cx = tmp_reg_coef[0], tmp_reg_coef[1]
        for edge in [('conditionP', 'all')]:
            for edge_ntw in [('phi','0.000500_caseDis0_chronic0')]:  # ,('RR', '0.001000'),('RR', '0.010000'), ('CC', '0.001000')
                # ('CC', '0.010000'),('cc0.001_and_phi0.001','_binary'),('cc0.001_and_phi0.001_and_rr0.001largerer2', '_binary')
                main(rpt, edge, edge_ntw, tmp_reg_coef_dx, tmp_reg_coef_cx, dir, col_year,dic_w)  # 用cx条件概率矩阵和(dx条件概率+DCN-phi0.0005的01值)矩阵正则;

    # param_tuning(edge=('conditionP', 'all'),edge_ntw=('CC', '0.001000'))



    # todo
    # 1. 加深stack
    # 2. 图卷积
    # 3. 邻接矩阵+度矩阵归一化
    # 4.第一层=邻接矩阵图学习，后几层=SA
    # 5. CC0.001和RR0.01融合


if __name__ == "__main__":
    main_()
