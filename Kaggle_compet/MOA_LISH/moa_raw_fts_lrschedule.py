# COLAB
folder_path = '/content/drive/MyDrive/Data/colabs_data/MOA_kaggle/'
from google.colab import drive
drive.mount('/content/drive')
!cp '/content/drive/MyDrive/Data/colabs_data/MOA_kaggle/quanvh8_funcs.py' .

# KAGGLE
# folder_path = '../input/lish-moa/'
# !cp '../input/coded-file/quanvh8_funcs.py' .


'''ENSEMBLE NETS
Inspire by https://www.kaggle.com/c/otto-group-product-classification-challenge/discussion/14335'''

import numpy as np, pandas as pd, copy, tensorflow as tf, matplotlib.pyplot as plt, sklearn

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

from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans

from quanvh8_funcs import (DerivedFeatures, kfolds_bagging_training, voting_predict,
                           kolds_stacked_ensemble_training, stacked_ensemble_predict )

import sys

def log_loss_metric(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    return bce(y_true, y_pred).numpy()

print(pd.__version__)
print(tf.__version__)


# Loading data and encoding
raw_test = pd.read_csv(folder_path + 'test_features.csv')
raw_train = pd.read_csv(folder_path + 'train_features.csv')
raw_targets = pd.read_csv(folder_path + 'train_targets_scored.csv')
raw_nonscored = pd.read_csv(folder_path + 'train_targets_nonscored.csv')

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

X_train = to_train[cols_fts]

# PARAMS
u_fts_num = X_train.shape[1]
i_fts_num = num_labels

initializer = tf.keras.initializers.LecunNormal()# 'he_normal'  # <-- update
kn_reg = tf.keras.regularizers.l1(2e-7) # <-- update
activation = 'selu'  # <-- update
bias_init_carefully = tf.keras.initializers.Constant(np.log([16844/(21948*206 - 16844)]))

# Define model
def layer_BDWD( n_components, activation = 'relu',  kn_init = 'glorot_uniform', kn_reg = None, bias_init = None):
  def layer_cpl(input_layer):
    '''BN - DROPOUT - WEIGHTNORMAL - DENSE'''
    layer = BatchNormalization() (input_layer)
    layer = Dropout(0.25 ) (layer)
    dense = Dense(n_components, activation = activation, 
                  kernel_initializer = initializer, kernel_regularizer = kn_reg ,
                  bias_initializer = bias_init)
    layer = WeightNormalization(dense) (layer)
    return layer
  return layer_cpl

# Model1 
def model1():
	input_u = Input(shape = (u_fts_num,) )
	layer_u = layer_BDWD(1024, activation = activation, kn_init = initializer, kn_reg = kn_reg, bias_init = bias_init_carefully) (input_u)
	layer_u = layer_BDWD(1024, activation = activation, kn_init = initializer, kn_reg = kn_reg, bias_init = bias_init_carefully) (layer_u)
	layer_u = layer_BDWD(512, activation = activation, kn_init = initializer, kn_reg = kn_reg, bias_init = bias_init_carefully) (layer_u)
	out_put = WeightNormalization(Dense(i_fts_num, activation = 'sigmoid' ))(layer_u)
	return Model(inputs=[input_u, ], outputs= [out_put])

model1 = model1()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

bce = tf.keras.losses.BinaryCrossentropy()
model1.compile(loss= BinaryCrossentropy(), optimizer=opt , metrics= [bce])


# Fitting
def exp_decay(lr0, s, down_hill):
    def exp_decay_fn(epoch):
        if epoch <= 5:
            out = lr0*2
        elif epoch%s <= 1:
            out = lr0
        else:
            out = lr0 + 0.015*(epoch%s/ (s - 1))* down_hill/ (down_hill + int(epoch/s) )
        return np.clip( out, 0.001, 0.05)
    return exp_decay_fn

fn_lr = exp_decay(0.001, 5, 50)
lr_schedule = tf.keras.callbacks.LearningRateScheduler ( fn_lr )

reduce_lr = ReduceLROnPlateau(monitor='val_binary_crossentropy', factor=0.1, patience=5, mode='min', min_lr=1E-5, verbose= 0)
early_stopping = EarlyStopping(monitor='val_binary_crossentropy', min_delta=1E-5, patience=15, mode='min',restore_best_weights=True, verbose= 0)
    
history = model1.fit(
        X_train, y_train, validation_split = 0.25, 
        callbacks=[early_stopping, lr_schedule], epochs=72, verbose =1,
        batch_size= 128)

# Graph___________________________________
hí = history.history
đầu_ra = {x: hí[x] for x in ['binary_crossentropy', 'val_binary_crossentropy']}
plt.plot(pd.DataFrame(đầu_ra).iloc[4:,:])
plt.show()
lr = []
for i in range(70) :
  lr.append( fn_lr(i+1) )
plt.plot(np.array(lr)[4:])
plt.show()