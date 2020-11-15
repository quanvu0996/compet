'''ENSEMBLE NETS
Inspire by https://www.kaggle.com/demetrypascal/fork-of-2heads-looper-super-puper-plate'''

import numpy as np
import pandas as pd
import copy

import tensorflow as tf
from tensorflow import feature_column as fc
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import (Dense, DenseFeatures, Dropout, 
                                     BatchNormalization, Embedding, Input, Concatenate, Average,
                                     InputLayer, Lambda)
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras import backend as K, Sequential, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop

import tensorflow_addons as tfa
from tensorflow_addons.layers import WeightNormalization
from keras.wrappers.scikit_learn import KerasRegressor
import keras

from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from math import log2

import sys
sys.path.append('../input/iterativestratification')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def log_loss_metric(y_true, y_pred):
    loss = 0
    y_pred_clip = np.clip(y_pred, 1e-15, 1 - 1e-15)
    for i in range(y_true.shape[1]):
        loss += - np.mean(y_true[:, i] * np.log(y_pred_clip[:, i]) + (1 - y_true[:, i]) * np.log(1 - y_pred_clip[:, i]))
    return loss / y_true.shape[1]

print(pd.__version__)
print(tf.__version__)

###############################################################################################################
# MODULE 1. DATA LOADING

# Loading data and encoding

folder_path = '../input/lish-moa/'
raw_test = pd.read_csv(folder_path + 'test_features.csv')
raw_train = pd.read_csv(folder_path + 'train_features.csv')
raw_targets = pd.read_csv(folder_path + 'train_targets_scored.csv')

# Phân loại dữ liệu
cols_id = ['sig_id']
cols_to_remove = ['cp_type']
cols_fts = [i for i in raw_train.columns if i not in cols_id +cols_to_remove]
cols_gene = [col for col in raw_train.columns if col.startswith("g-")]
cols_cell = [col for col in raw_train.columns if col.startswith("c-")]
cols_experiment = [col for col in cols_fts if col not in cols_gene+cols_cell]
cols_target = [i for i in raw_targets.columns if i not in cols_id]
num_fts, num_labels = len(cols_fts), len(cols_target)

# xử lý categorical
def transform_data(input_data):
    '''Clean data and encoding
        * input_data: table '''
    out = input_data.copy()
    out['cp_dose'] = out['cp_dose'].map({'D1':0, 'D2':1})
    out['cp_time'] = out['cp_time']/72
    
    return out

to_train = transform_data(raw_train[raw_train['cp_type'] != 'ctl_vehicle'])
to_train_targets = raw_targets.iloc[to_train.index]
full_pred  = transform_data(raw_test)
to_pred = full_pred[full_pred['cp_type'] != 'ctl_vehicle']

###############################################################################################################
# MODULE 2. DATA TRANSFORMATION PIPELINE

# Add derived fts
class DerivedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters= 6):
        self.n_clusters = n_clusters

    def fit(self, X, y= None):
        try: X_ = X.values.copy() # creating a copy to avoid changes to original dataset
        except: X_ = X.copy()

        # try: y_ = y.values.copy()
        # except: y_ = y.copy()

        # self.n_labels = y_.shape[1]

        # Kmeans
        self.kmeans = KMeans(n_clusters= self.n_clusters ) 
        self.kmeans.fit(X_)

        # self.closet_labels = y_[np.argmin(self.kmeans.transform(X_), axis= 0)]# closet labels
        return self

    def transform(self, X, y = None):
        try: X_ = X.values.copy() # creating a copy to avoid changes to original dataset
        except: X_ = X.copy()

        # Non-linear infomation
        std = np.std(X_, axis= 1)

        # Kmeans features
        distance_to_centroid = self.kmeans.transform(X_) # distance to the centroid
        cluster_labels = self.kmeans.predict(X_)# cluster index

        # column names
        self.columns = ['std']\
                        + ['distance_to_centroid'+str(i) for i in range(distance_to_centroid.shape[1])]\
                        + ['cluster_labels']
        ouput = np.concatenate([std.reshape(-1,1), distance_to_centroid, cluster_labels.reshape(-1,1)], axis= 1)
        return ouput

# preprocessing pipeline
def pipe_line_builder(quantiles_num, pca_dims, kmean_clusters):
    '''Dựng pipe line cho từng nhóm columns
    :quantiles_num: int: số quantile khi normalise
    :pca_dims: int: số chiều pca'''
    norm = QuantileTransformer(n_quantiles=quantiles_num,random_state=0, output_distribution="normal")
    pca = PCA(n_components = pca_dims)
    derived_ft = DerivedFeatures(n_clusters = kmean_clusters)

    p_derived_ft = Pipeline([
        ('norm', norm), 
        ('derived', derived_ft)])

    p_norm_pca = Pipeline([ 
        ('norm', norm),
        ('pca', pca) ])
    return FeatureUnion([
        ('norm', norm), 
        ('norm_pca', p_norm_pca),
        ('derived', p_derived_ft)])

# 

pipe = Pipeline([
    ('norm_pca', ColumnTransformer([
                     ('gene', pipe_line_builder(quantiles_num = 200, pca_dims = 600, kmean_clusters = 5), cols_gene),
                     ('cell', pipe_line_builder(quantiles_num = 200, pca_dims = 50, kmean_clusters = 5), cols_cell),
                    ]) 
    ), 
    ('var', VarianceThreshold(0.02)) 
])

pipe = ColumnTransformer([
    ('gene_cell', pipe, cols_gene+ cols_cell),
    ('experiment', 'passthrough', cols_experiment)
])

# Transform data
pipe.fit(to_train[cols_fts].append(to_pred[cols_fts]))
X_train = pipe.transform(to_train[cols_fts])
X_pred = pipe.transform(to_pred[cols_fts])
y_train = to_train_targets[cols_target]

###############################################################################################################
# MODULE 3: BAGGING

# Random bagging nets
def bagging_split(X_train, y_train, alpha, n_samples):
    ''' SPLIT TRAINING DATA TO N SAMPLES FOR BAGGING
    :X_train:np array: data for model training only
    :y_train:np array: labels for model training only
    :alpha:0-1 float: poportion of data in each sample
    return:
        generator for bagging training set'''
    data_length = X_train.shape[0]
    for i in range(n_samples):
        idx = np.random.choice( data_length, size= int(data_length * alpha), replace=0)
        yield X_train[idx], y_train[idx]

def voting_predict(model_list, X_pred, num_labels):
    '''PREDICT OUTPUT FROM A LIST OF MODEL'''
    pred = np.zeros((X_pred.shape[0], num_labels))
    for model in model_list:
        pred_i = model.predict(X_pred)
        pred += pred_i
    return pred/len(model)


def bagging_training(model, X_train, y_train, X_val, y_val, alpha, n_samples):
    '''TRAINING FOR EACH BOOSTRAP AGGREGATING (BAGGING)
    return:
        list of n_samples model'''
    ouput = []
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, mode='min', min_lr=1E-5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=1E-5, patience=15, mode='min',restore_best_weights=True, verbose=1)

    for bag_id, (x_train_bag, y_train_bag) in enumerate(bagging_split(X_train, y_train, alpha = alpha, n_samples = n_samples)):
        tf.keras.backend.clear_session()
        tf.random.set_seed(np.random.random())
        print('Training at Bag ', bag_id, '_'*100)

        model_ = copy.deepcopy(model)
        model_.fit(
            x_train_bag, y_train_bag,  validation_data = (X_val, y_val), callbacks=[reduce_lr, early_stopping], epochs=150,
            batch_size=BATCH_SIZE )
        ouput.append(model_)
        print('Logloss at bag ', bag_id, ': ', log_loss_metric(y_val, model_.predict(X_val)))
        del model_
    logloss_all_bag = log_loss_metric(y_val, 
                voting_predict(ouput, X_val, num_labels = y_val[1])  )
    print('Evaluate bagging, log loss = ', logloss_all_bag )

    return ouput


# Kfolds evaluation
def kfolds_training(NFOLDS, model, SEEDS, X_train, y_train, bagging_alpha = 0.75, bagging_samples = 10):
    ''' TRAINING FOR KFOLDS EVALUATION
    :NFOLDS:int: số folds
    :model:model: model dùng để train
    :SEEDS:list: list of seeds to train
    :X_train:np array: full data for train and evaluate
    :y_train:np array: full labels for train and evaluate
    return:
        :list of list model: list NFOLDS-list trained model
    '''
    ouput = []
    kf = KFold(n_splits= NFOLDS, shuffle = True)

    for fols_id, (train_index, val_index) in enumerate(kf.split(to_train)):
        print('Training at fold: ', fols_id, '#'*100)
        tf.keras.backend.clear_session()

        fold_X_train, fold_y_train = X_train[train_index], y_train[train_index]
        fold_X_val, fold_y_val = X_train[val_index], y_train[val_index]

        # Training bagging
        model_list = bagging_training(model, fold_X_train, fold_y_train, fold_X_val ,fold_y_val,
            alpha = bagging_alpha, n_samples = bagging_samples)

        # fold_logloss = log_loss_metric(fold_y_val, 
        #     voting_predict( model_list, fold_X_val, 206))
        ouput.append(model_list)
    fold_logloss = log_loss_metric(fold_y_val, 
            voting_predict( sum(ouput, []), fold_X_val, 206))
    print('AVG logloss all folds: ', fold_logloss)

    return ouput

# Hyper params
NFOLDS = 7
BATCH_SIZE = 128
EPOCHS = 150
BAGGING_ALPHA = 0.75
SEEDS = [23, 228, 1488, 1998, 2208, 2077, 404]
KFOLDS = 10
label_smoothing_alpha = 0.0005
P_MIN = label_smoothing_alpha
P_MAX = 1 - P_MIN

# Define model
model = Sequential([
    BatchNormalization(),
    WeightNormalization(Dense(1024, activation="relu")),
    BatchNormalization(),
    Dropout(0.25),
    WeightNormalization(Dense(512, activation="relu")),
    BatchNormalization(),
    Dropout(0.25),
    WeightNormalization(Dense(256, activation="relu")),
    BatchNormalization(),
    Dropout(0.25),
    WeightNormalization(Dense(num_labels, activation="sigmoid"))
])

def logloss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred,P_MIN,P_MAX)
    return -K.mean(y_true*K.log(y_pred) + (1-y_true)*K.log(1-y_pred))

model.compile(optimizer='adam', loss=BinaryCrossentropy(label_smoothing= label_smoothing_alpha), metrics=logloss)


model_2list = kfolds_training(NFOLDS, model, SEEDS, X_train, y_train, bagging_alpha = 0.75, bagging_samples = 10)

prediction = voting_predict( sum(model_2list, []), X_pred, 206)

    
    
    
    
    
    ss += model.predict(pipe.transform(to_pred_non_ctl))

ss = ss/NFOLDS


###################################################################
# TỐI ƯU KIỂU MẠNG
#1. User embedding
#1.1. Gene fts
input_g = Input(shape = (g_ft_num,) )
layer_g = WeightNormalization(Dense( 512, activation = 'elu', kernel_initializer='he_normal')) (input_g)
layer_g = Dropout(0.2619422201258426) (layer_g)
layer_g = BatchNormalization() (layer_g)

layer_g = WeightNormalization(Dense( 320, activation = 'elu', kernel_initializer='he_normal')) (layer_g)
layer_g = Dropout(0.2619422201258426) (layer_g)
layer_g = BatchNormalization() (layer_g)

#1.2. Cell fts
input_c = Input(shape = (c_ft_num,) )
layer_c = WeightNormalization(Dense( 80, activation = 'elu', kernel_initializer='he_normal')) (input_c)
layer_c = Dropout(0.2619422201258426) (layer_c)
layer_c = BatchNormalization() (layer_c)

#1.3. Experiment fts
layer_e = Input(shape = (e_ft_num,) )

#1.4 user full fts with residual connection
layer_u = Concatenate() ([layer_g,input_g, layer_c,input_c, layer_e])

layer_u = WeightNormalization(Dense( n_components*2, activation = 'elu', kernel_initializer='he_normal')) (layer_u)
layer_u = Dropout(0.2619422201258426) (layer_u)
layer_u = BatchNormalization() (layer_u)

layer_u = WeightNormalization(Dense( n_components, activation = 'elu', kernel_initializer='he_normal')) (layer_u)
layer_u = Dropout(0.2619422201258426) (layer_u)
layer_u = BatchNormalization() (layer_u)



#2. Item embedding
#2.1. Addition information for item_info
chemical_category = tf.transpose(
        tf.constant(
            [[1 if '_inhibitor' in i else 0 for i in cols_target],
               [1 if '_agonist' in i else 0 for i in cols_target],
               [1 if '_agent' in i else 0 for i in cols_target],
               [1 if '_antagonist' in i else 0 for i in cols_target],
               [1 if '_blocker' in i else 0 for i in cols_target],
               [1 if '_activator' in i else 0 for i in cols_target] 
             ]))

#2.2 Full item fts: addition + onehot
item_ft = tf.concat(
    [chemical_category ,
     tf.eye(i_fts_num, dtype = tf.int32) # Create tensor 0-1 coresponse with chemical labels
    ], axis = 1
)
layer_i = Dense(n_components, activation = 'relu', kernel_initializer='he_normal', name ='layer_u1') (item_ft)


#3. Dot product user - item
def dot_2layer(x):
    return K.dot( x[0], K.transpose(x[1]))
dot_ui = Lambda( dot_2layer, name = 'lambda_dot' ) ([layer_u,layer_i])
dot_ui= WeightNormalization(Dense(512, activation="relu", kernel_initializer='he_normal')) (dot_ui)
dot_ui= BatchNormalization() (dot_ui)
dot_ui = WeightNormalization(Dense(i_fts_num, activation = 'sigmoid', kernel_initializer='he_normal', name = 'labels'))(dot_ui)

# Compile model
model = Model(inputs=[layer_e, input_g, input_c, ], outputs= [dot_ui])

step = tf.Variable(0, trainable=False)
schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
    [10000, 15000], [1e-0, 1e-1, 1e-2])
lr = 1e-1 * schedule(step)
wd = lambda: 1e-3 * schedule(step)
opt = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)

model.compile(loss= BinaryCrossentropy(label_smoothing=0.0005), optimizer='adam')
print( model.summary() )

----------------------------------------------------------------------------------------------

layer_u = Sequential([
    BatchNormalization(),
    WeightNormalization(Dense(1024, activation="selu")),
    BatchNormalization(),
    Dropout(0.25),
    WeightNormalization(Dense(512, activation="selu")),
    BatchNormalization(),
    Dropout(0.25),
    WeightNormalization(Dense(256, activation="selu")),
    BatchNormalization(),
    Dropout(0.25),
    WeightNormalization(Dense(num_labels, activation="sigmoid"))
])

#2.1. Addition information for item_info
chemical_category = tf.transpose(
        tf.constant(
            [[1 if '_inhibitor' in i else 0 for i in cols_target],
               [1 if '_agonist' in i else 0 for i in cols_target],
               [1 if '_agent' in i else 0 for i in cols_target],
               [1 if '_antagonist' in i else 0 for i in cols_target],
               [1 if '_blocker' in i else 0 for i in cols_target],
               [1 if '_activator' in i else 0 for i in cols_target] 
             ]))

#2.2 Full item fts: addition + onehot
item_ft = tf.concat(
    [chemical_category ,
     tf.eye(i_fts_num, dtype = tf.int32) # Create tensor 0-1 coresponse with chemical labels
    ], axis = 1
)
layer_i = Dense(n_components, activation = 'relu', kernel_initializer='he_normal', name ='layer_u1') (item_ft)


#3. Dot product user - item
def dot_2layer(x):
    return K.dot( x[0], K.transpose(x[1]))
dot_ui = Lambda( dot_2layer, name = 'lambda_dot' ) ([layer_u,layer_i])
dot_ui= WeightNormalization(Dense(512, activation="selu", kernel_initializer='he_normal')) (dot_ui)
dot_ui= BatchNormalization() (dot_ui)
dot_ui= Dropout(0.25) (dot_ui),
dot_ui = WeightNormalization(Dense(i_fts_num, activation = 'sigmoid', kernel_initializer='he_normal', name = 'labels'))(dot_ui)

model = Model(inputs=[ layer_u, ], outputs= [dot_ui])