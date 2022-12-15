from copy import deepcopy
import warnings
warnings.filterwarnings('always')
warnings.simplefilter('always')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import glob, os, sys
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import csv
import pickle
import math
import os
import traceback
from datetime import datetime
from numpy import random
import statistics
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator


from Util import Util

class VibrationML:
    def __init__(self, model_name, model_path=None):
        self.model_name = model_name
        self.all_training_dataset_windows = [] # list of tuple (X_train, y_train)
        self.autoscaler = None

        if(self.model_name == "LR"):
            if(model_path is None):
                Util.create_directory("output")
                self.model_check_point_path = os.path.join("output", "LR.pkl")
                self.autoscaler_check_point_path = os.path.join("output", "autoscaler.pkl")
            else:
                self.model_check_point_path = model_path

        if(self.model_name == "RF"):
            if(model_path is None):
                Util.create_directory("output")
                self.model_check_point_path = os.path.join("output", "RF.pkl")
                self.autoscaler_check_point_path = os.path.join("output", "autoscaler.pkl")
            else:
                self.model_check_point_path = model_path

        if(self.model_name == "LSTM"):
            if(model_path is None):
                self.model_check_point_path = os.path.join("output", "LSTM.h5")
                self.autoscaler_check_point_path = os.path.join("output", "autoscaler.pkl")
            else:
                self.model_check_point_path = model_path
            self.time_steps = 10

        if(self.model_name == "LSTM_Forecast"):
            if(model_path is None):
                self.model_check_point_path = os.path.join("output", "LSTM_Forecast.h5")
                self.autoscaler_check_point_path = os.path.join("output", "autoscaler.pkl")
            else:
                self.model_check_point_path = model_path
            self.look_back = 10

        
        self.model = self.get_model()
    
    def add_new_training_data_window(self, new_data_window):
        # new_data: contains a window of data samples
        self.all_training_dataset_windows.append(new_data_window)

    def clear_training_data_window(self):
        self.all_training_dataset_windows = []

    def get_training_dataset(self, prev_window=None):
        if prev_window == None:
            prev_window = len(self.all_training_dataset_windows)
        
        training_data_windows = self.all_training_dataset_windows[-prev_window:]
        # print(type(training_data), len(training_data))
        # print(training_data[0])
        X = []
        Y = []
        for data_window in training_data_windows:
            for data in data_window:
                (x, y) = data
                X.append(x)
                Y.append(y)
        # print(len(X), len(training_data))
        # X_train = pd.concat(X)
        # y_train = pd.concat(Y)
        X_train = pd.DataFrame(X)
        y_train = Y
        # print(len(X_train), len(y_train))
        return (X_train, y_train)


    
    def get_model(self):
        if(self.model_name == "LR"):
            return LinearRegression(normalize=True)

        if(self.model_name == "RF"):
            return RandomForestRegressor()

        if(self.model_name == "LSTM"):
            lstm_model = tf.keras.models.Sequential([
                # Shape [batch, time, features] => [batch, time, lstm_units]
                tf.keras.layers.LSTM(32, return_sequences=False),
                # Shape => [batch, time, features]
                tf.keras.layers.Dense(units=1)
            ])
            return lstm_model

        if(self.model_name == "LSTM_Forecast"):
            model = tf.keras.models.Sequential()
            model.add(
                tf.keras.layers.LSTM(64,
                    input_shape=(self.look_back, 1))
            )
            model.add(tf.keras.layers.Dense(1))
            return model


    def does_have_pretrained_weights(self):
        model = self.load_model_weights()
        return model != None

    def load_model_weights(self, model_save_path=None):
        if(model_save_path is not None):
            self.model_check_point_path = model_save_path
        model = None
        if(self.model_name == "LR"):
            if os.path.exists(self.model_check_point_path):
                model = Util.static_load_obj(self.model_check_point_path)         

        if(self.model_name == "RF"):
            if os.path.exists(self.model_check_point_path):
                model =  Util.static_load_obj(self.model_check_point_path)

        if(self.model_name == "LSTM"):
            if os.path.exists(self.model_check_point_path):
                model = tf.keras.models.load_model(self.model_check_point_path)

        return model

    def save_model_weights(self, model_save_path=None):
        if(model_save_path is not None):
            self.model_check_point_path = model_save_path
        if(self.model_name == "LR"):
            Util.static_save_obj(self.model, self.model_check_point_path)
        if(self.model_name == "RF"):
            Util.static_save_obj(self.model, self.model_check_point_path)
        # if(self.model_name == "LSTM"):
        #     Util.static_save_obj(self.model, self.model_check_point_path)


    def finetune_model(self, prev_data_windows=None):
        X_train, y_train = self.get_training_dataset(prev_window=prev_data_windows)
        print(type(X_train))
        print(X_train)

        if(self.model_name == "LSTM_Forecast"):
            y_train = y_train.tolist()
            # print(len(y_train), type(y_train), y_train)
            train_dataset = TimeseriesGenerator(y_train, y_train, length=self.look_back, batch_size=32)
            self.LSTM_compile_and_fit(train_dataset)
            return

        X_train_normalized = self.get_normalized_train_data(X_train)

        if(self.model_name == "LR" or self.model_name=="RF"):
            self.model.fit(X_train_normalized, y_train)
            self.save_model_weights()
        
        if(self.model_name == "LSTM"):
            train_dataset = TimeSeriesData.make_train_dataset(X_train_normalized, y_train,  self.time_steps)
            self.LSTM_compile_and_fit(train_dataset)





    def make_prediction(self, X_test):
        X_test_df = pd.DataFrame(X_test)
        X_test_normalized = self.get_normalized_test_data(X_test_df)
        if(self.model_name == "LR" or self.model_name=="RF"):
            y_pred = self.model.predict(X_test_normalized)
        if(self.model_name == "LSTM"):
            test_ds = TimeSeriesData.make_test_dataset(X_test_normalized, self.time_steps)
            y_pred = self.model.predict(test_ds)
            y_pred = np.clip(y_pred, 0, max(y_pred))
            y_pred = ([0] * (self.time_steps-1)) + [p[0] for p in y_pred]
            y_pred = np.array(y_pred)

        if(self.model_name == "LSTM_Forecast"):
            y_pred = []
            if(len(self.all_training_dataset_windows) > 0):
                test_sample = self.all_training_dataset_windows[-1][1].tolist()
            while (len(y_pred) < len(X_test)):
                test_sample = test_sample[-(self.look_back+1):]
                # print(type(test_sample), len(test_sample), test_sample)
                
                test_sample = list(Util.get_smooth_data(test_sample, N=2))
                # print(type(test_sample), len(test_sample), test_sample)
                
                
                test_gen = TimeseriesGenerator(test_sample, test_sample, length=self.look_back)
                pred = self.model.predict(test_gen)[0][0]
                # print("Test Sample len: %d" % len(test_sample))
                y_pred.append(pred)
                test_sample.append(pred)
            y_pred = np.clip(y_pred, 0, max(y_pred))
            # print(y_pred)
        return list(y_pred)

    def get_max_min_range(self, y_pred):
        # Needs to update the heuristics on this model
        X, y = self.get_training_dataset(prev_window=3)
        prev_SD = statistics.stdev(y)

        y_pred_max = [ d + random.uniform(prev_SD, 3 * prev_SD) for d in y_pred]
        y_pred_min = [ max(0, d - random.uniform(prev_SD, 3 * prev_SD)) for d in y_pred]
        return y_pred_max, y_pred_min


    def get_normalized_train_data(self, X_train):
        X_train = X_train.copy(deep=True)
        self.autoscaler = StandardScaler()
        self.autoscaler.fit(X_train)
        # X_train[Util.FEATURES] = self.autoscaler.transform(X_train[Util.FEATURES])
        X_train = self.autoscaler.transform(X_train)
        Util.static_save_obj(self.autoscaler, self.autoscaler_check_point_path)
        # X_test[feature_columns] = autoscaler.transform(X_test[feature_columns])
        return X_train

    def get_normalized_test_data(self, X_test):
        X_test = X_test.copy(deep=True)
        if self.autoscaler is None:
            self.autoscaler = Util.static_load_obj(self.autoscaler_check_point_path)
        X_test = self.autoscaler.transform(X_test)
        # X_test[Util.FEATURES] = self.autoscaler.transform(X_test[Util.FEATURES])
        return X_test
        
    def LSTM_compile_and_fit(self, train_ds, max_epochs = 100, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                            patience=patience,
                                                            mode='min')
                                                          
        monitor_it = tf.keras.callbacks.ModelCheckpoint(self.model_check_point_path, monitor='loss',\
                                                    verbose=0, save_best_only=True,\
                                                    save_weights_only=False,\
                                                    mode='min')

        self.model.compile(loss=tf.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam(),
                        # metrics=[tf.metrics.MeanSquaredError(), tf.metrics.MeanSquaredLogarithmicError()],
                        run_eagerly=True
                        )

        history = self.model.fit(train_ds, epochs=max_epochs, callbacks=[early_stopping, monitor_it], verbose=0)
        return history


class TimeSeriesData():

    def make_train_dataset(X, y, time_steps, batch_size=32, ):
        data = np.array(X, dtype=np.float32)
        target = np.array(y, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(data=data, 
                                                          targets=target, 
                                                          sequence_length=time_steps, 
                                                          sequence_stride=1, 
                                                          shuffle=True, 
                                                          batch_size=batch_size)
        return ds



    def make_test_dataset(X, time_steps, batch_size=32):
        data = np.array(X, dtype=np.float32)
        # target = np.array(y, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(data=data, 
                                                          targets=None, 
                                                          sequence_length=time_steps, 
                                                          sequence_stride=1, 
                                                          shuffle=False, 
                                                          batch_size=batch_size)
        return ds





