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

reduce_lr = ReduceLROnPlateau(monitor='val_binary_crossentropy', factor=0.3, patience=5, mode='min', min_lr=1E-5, verbose=0 )
early_stopping = EarlyStopping(monitor='val_binary_crossentropy', min_delta=1E-5, patience=15, mode='min',restore_best_weights=True, verbose=0 )

# Log_loss return numpy values
def log_loss_metric(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    return bce(y_true, y_pred).numpy()

# Add derived fts
class DerivedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters= 6):
        self.n_clusters = n_clusters

    def fit(self, X, y= None):
        try: X_ = X.values.copy() # creating a copy to avoid changes to original dataset
        except: X_ = X.copy()

        # Kmeans
        self.kmeans = KMeans(n_clusters= self.n_clusters ) 
        self.kmeans.fit(X_)

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

# Predict with average ouput from a list of models
def voting_predict(model_list, X_pred):
    '''PREDICT OUTPUT FROM A LIST OF MODEL'''
    pred = 0
    for model in model_list:
        pred += model.predict(X_pred)
    avg_pred = pred/len(model_list)
    avg_pred[:,[34,82]] = 0
    return avg_pred

# Boosting prediction
def boosting_predict(model_list, X_pred):
    '''model_list: danh sách các model dự đoán sai số, model sau dự đoán sai số của model trước, dự đoán ban đầu =0'''
    y_pred = 0
    for model in model_list:
        error_pred = model.predict(X_pred)
        y_pred += error_pred
    return y_pred


def reset_seed_model(model):
    seed = int( np.random.random() *100)
    np.random.seed(seed)
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    try:
        model_ = copy.deepcopy(model)
    except:
        model_ = copy.copy(model)
    return model_


# Boosting function
def boosting_training(model, X_train, y_train, X_val, y_val, n_interations = 5):
    '''Training boosting model'''
    pred = 0
    error = y_train - pred
    model_list = []

    for i in range(n_interations):
        model_ = reset_seed_model(model)
        model_.fit(
            X_train, error, validation_data = (X_val, y_val), 
            callbacks=[reduce_lr, early_stopping], epochs=150, verbose = 0,
            batch_size=BATCH_SIZE )
        
        error_pred = model_.predict(X_train)
        pred += error_pred 
        error = y_train - pred

        model_list.append(model_)
        del model_

    print('Logloss: ', log_loss_metric(y_val, boosting_predict(model_list, X_val)))
    return model_list


def bagging_training(model, X_train, y_train, X_val, y_val, alpha, n_samples, save_path = '', callbacks = [], epochs = 47, batch_size = 128):
    '''TRAINING FOR EACH BOOSTRAP AGGREGATING (BAGGING)
    return:
        list of n_samples model'''
    model_list = []

    for bag_id, (x_train_bag, y_train_bag) in enumerate(bagging_split(X_train, y_train, alpha = alpha, n_samples = n_samples)):
        model_ = reset_seed_model(model)
        model_.fit(
            x_train_bag, y_train_bag, validation_data = (X_val, y_val), 
            callbacks= callbacks , epochs= epochs, verbose = 0,
            batch_size= batch_size )
        
        y_pred_inbag = model_.predict(X_val)
        evaluate_at_bag = log_loss_metric(y_val, y_pred_inbag)
        print('Logloss at bag ', bag_id, ': ', evaluate_at_bag)
        model_list.append(model_)
        del model_
    
    print('Evaluate bagging, log loss = ', 
        log_loss_metric(y_val,  voting_predict(model_list, X_val, num_labels = y_val.shape[1])  ) )

    return model_list



def kfolds_bagging_training(NFOLDS, model, SEEDS, X_train, y_train, bagging_alpha = 0.75, bagging_samples = 10, callbacks = [], epochs = 47, batch_size = 128):
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
    fold_logloss = []

    for fold_id, (train_index, val_index) in enumerate(kf.split(X_train)):
        print('Training at fold: ', fold_id, '-'*100)

        fold_X_train, fold_y_train = X_train[train_index], y_train[train_index]
        fold_X_val, fold_y_val = X_train[val_index], y_train[val_index]

        # Training bagging
        model_list = bagging_training(model, fold_X_train, fold_y_train, fold_X_val ,fold_y_val,
            alpha = bagging_alpha, n_samples = bagging_samples, callbacks = callbacks, epochs = epochs, batch_size = batch_size)

        fold_logloss.append( log_loss_metric(fold_y_val,  voting_predict( model_list, fold_X_val, 206)) )
        ouput.append(model_list)
    
    print('AVG logloss all folds: ', np.mean(fold_logloss))

    return ouput

def kfolds_boosting_training(NFOLDS, model, SEEDS, X_train, y_train, n_interations = 5):
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
    fold_logloss = []

    for fold_id, (train_index, val_index) in enumerate(kf.split(to_train)):
        print('Training at fold: ', fold_id, '#'*100)

        fold_X_train, fold_y_train, fold_X_val, fold_y_val = X_train[train_index], y_train[train_index], X_train[val_index], y_train[val_index]

        # Training bagging
        model_list = boosting_training( model, fold_X_train, fold_y_train, fold_X_val ,fold_y_val,
                                        n_interations)

        fold_logloss.append( log_loss_metric(fold_y_val,  voting_predict( model_list, fold_X_val, 206)) )
        ouput.append(model_list)
    
    print('AVG logloss all folds: ', np.mean(fold_logloss))

    return ouput


# Training với một list model đầu vào không nhất thiết cùng 1 thuật toán, nhưng phải có chung cách train là .fit
def multi_bagging_training(model_list, X_train, y_train, X_val, y_val, alpha = 0.75, n_interations = 2):
    '''Training nhiều model để chạy ensemble
    :model_list:list of tuple: list các cấu hình model có dạng: model - loại model - tham số
                    [(model_1, 'include_valid_set', [epochs = 10, batch_size])]
    return:
        list models với số model = len(model_list) * n_interations '''
    trained_models = []

    for bag_id, (x_train_bag, y_train_bag) in enumerate(bagging_split(X_train, y_train, alpha = alpha, n_samples = n_interations)):
        for model_inf in model_list:
            model = model_inf[0]
            model_type = model_inf[1]
            model_params = model_inf[2]

            model = reset_seed_model(model)
            if model_inf[1] == 'include_valid_set':         
                model.fit( x_train_bag, y_train_bag, validation_data = (X_val, y_val), *model_params )
            else :
                model.fit( x_train_bag, y_train_bag, *model_params )

            trained_models.append(model)
    return trained_models

# Các tầng ensemble
def stacked_ensemble_training( stacked_models, X_train, y_train, X_val, y_val, alpha = 0.75, n_interations = 2):
    '''
    Dùng first_layer_models để fit với dữ liệu đầu vào
    Dùng second_layers_models để quyết định ensemble kết quả đầu ra của first_layer_model_list như thế nào '''
    stacked_trained_models = []
    meta_train_fts = X_train
    meta_val_fts = X_val

    for i in range(len(stacked_models)):
        trained_models = multi_bagging_training(stacked_models[i], meta_train_fts, y_train, meta_val_fts, y_val, alpha = alpha, n_interations = n_interations)
        stacked_trained_models.append(trained_models)

        meta_train_fts =  np.concatenate(   [ model.predict(meta_train_fts) for model in trained_models ] , axis = 1)
        # meta_val_fts =  np.concatenate(   [ model.predict(meta_val_fts) for model in trained_models ] , axis = 1)
 
    return stacked_trained_models

def stacked_ensemble_predict( stacked_models, X_pred ):
    ''' Chạy kết quả predict cho 1 stacked_ensemble_models có 2 layers, không sử dụng weight
    :stacked_models:tuple 2 layers model '''
    meta_fts = X_pred

    for i in range( len(stacked_models) ):
        prediction_list = [ model.predict(meta_fts) for model in stacked_models[0] ]
        meta_fts = np.concatenate( prediction_list , axis = 1)

    meta_fts = 0 # Giải phóng bộ nhớ
    return np.mean(prediction_list, axis = 0)


def kolds_stacked_ensemble_training(NFOLDS, stacked_models, X_train, y_train, alpha = 0.75, n_interations = 2):
    kf = KFold(n_splits= NFOLDS, shuffle = True)
    fold_logloss = []

    for fold_id, (train_index, val_index) in enumerate(kf.split(X_train)):
        print('Training at fold: ', fold_id, '#'*100)

        fold_X_train, fold_y_train, fold_X_val, fold_y_val = X_train[train_index], y_train[train_index], X_train[val_index], y_train[val_index]

        # Training bagging
        stacked_trained_models = stacked_ensemble_training(stacked_models, fold_X_train, fold_y_train, fold_X_val, fold_y_val, alpha, n_interations)

        predictions = stacked_ensemble_predict(stacked_trained_models, fold_X_val)

        fold_log_loss = log_loss_metric(fold_y_val,  predictions)
        print('Logloss at fold ', fold_id, ' : ', fold_log_loss)
        fold_logloss.append( fold_log_loss )
    
    print('AVG logloss all folds: ', np.mean(fold_logloss))

    return stacked_trained_models


def kfolds_pnf_training(NFOLDS, models, X_train, y_train):
    ''' TRAINING FOR KFOLDS EVALUATION - predict and fix model
    :NFOLDS:int: số folds
    :models:list: predictor + fixer
    :SEEDS:list: list of seeds to train
    :X_train:np array: full data for train and evaluate
    :y_train:np array: full labels for train and evaluate
    return:
        :list of list model: list NFOLDS-list trained model
    '''
    kf = KFold(n_splits= NFOLDS, shuffle = True)
    fold_logloss = []


    for fold_id, (train_index, val_index) in enumerate(kf.split(X_train)):
        print('Training at fold: ', fold_id, '#'*100)

        fold_X_train, fold_y_train, fold_X_val, fold_y_val = X_train[train_index], y_train[train_index], X_train[val_index], y_train[val_index]

        # Training bagging
        predictor = reset_seed_model(models[0]) 
        fixer = reset_seed_model(models[1]) 

        reduce_lr = ReduceLROnPlateau(monitor='val_binary_crossentropy', factor=0.3, patience=5, mode='min', min_lr=1E-5, verbose=1)
        early_stopping = EarlyStopping(monitor='val_binary_crossentropy', min_delta=1E-5, patience=15, mode='min',restore_best_weights=True, verbose=1)
            
        predictor.fit(  fold_X_train, fold_y_train, validation_data = (fold_X_val, fold_y_val), 
                        callbacks=[reduce_lr, early_stopping], epochs=150, verbose =0, batch_size=128   )
        fixer.fit( fold_X_train, fold_y_train - predictor.predict(  fold_X_train ) )
        
        model_list = [predictor, fixer]
        fold_logloss.append( log_loss_metric(fold_y_val,  boosting_predict( model_list, fold_X_val)) )
        print('Val logloss at fold ', fold_id, ' : ', fold_logloss[fold_id])
    
    print('AVG logloss all folds: ', np.mean(fold_logloss))

    return model_list