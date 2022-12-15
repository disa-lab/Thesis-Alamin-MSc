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

try:
    import matplotlib.pylab as plt
except ImportError:
    traceback.print_exc()

import vibration_prediction.Vibration_Model as VP_model
import vibration_prediction.Util as Util

def test_load():
    print("Vibration model loaded properly 2")


def static_load_obj(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def static_save_obj(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    # print("Saved object to a file: %s" % (str(file_path)))

def get_smooth_data(data, N):
    ret = np.cumsum(data, dtype=float)
    ret[N:] = ret[N:] - ret[:-N]
    ret_avg =  ret[N - 1:] / N
    ret = np.concatenate([data[:N-1], ret_avg])
    assert len(data) == len(ret)
    return ret

FEATURES = ["ROP", "Torque", "Weight_on_Bit"]
TARGET = "Accel"
DT = "Date_Time"

class Dataset:
    def __init__(self, data_file, model_name):
        self.data_file = data_file
        self.df_dataset = pd.read_csv(data_file)
        self.cur_data_window_ind = 0
        self.model = VP_model.Model(model_name)

    def generate_result_for_run(self, run):
        self.setup_run(run=run)
        self.generate_prediction_runwise()
        self.visualize_prediction_runwise()

    def setup_run(self, run):
        self.run = run       
        self.df_run = self.df_dataset[self.df_dataset['Run'] == run]
        self.df_run_time_bin = self.classify_to_bins(self.df_run, bin_type="time", bin_size=10) # 10 min
        self.bins = self.df_run_time_bin['Bins'].unique()
        self.test_bin_time_stamps = self.get_bin_stamps(self.df_run_time_bin)
        self.data_windows = self.get_dataset_windows()
        self.df_overall_run_predict = None

    # def more_data_window(self):
    #     return  self.cur_data_window_ind < len(self.data_windows)
    
    # def next_data_window(self):
    #     data_window = self.data_windows[self.cur_data_window_ind]
    #     self.cur_data_window_ind = self.cur_data_window_ind + 1
    #     return data_window

    def get_dataset_windows(self):
        data_windows = []
        # print(self.df_run_time_bin.columns)
        for test_bin in self.bins:
            df =  self.df_run_time_bin[self.df_run_time_bin['Bins'] == test_bin]
            X = df[FEATURES]
            y = df[TARGET]
            t = df[DT]
            # print(len(X), len(y))
            data = (X, y, t)
            data_windows.append(data)
        return data_windows

    def generate_prediction_runwise(self):
        # print(len(dataset.data_windows))
        
        self.cur_data_window_ind = 0
        
        if(self.model.does_have_pretrained_weights() is False or len(self.model.all_training_dataset_windows) == 0):
            for i in range(3):
                self.model.add_new_training_data_window(self.data_windows[self.cur_data_window_ind])
                self.cur_data_window_ind = self.cur_data_window_ind + 1
                # print("Skipping testing on this bin: %d" % self.cur_data_window_ind)
            self.model.finetune_model()
            # self.model.model = self.model.load_model_weights()
        else:
            self.model.model = self.model.load_model_weights()
        # for i in range(3):
        #     LR.add_new_training_data_window(dataset.data_windows[i])
        

        df_overall_run_predict = pd.DataFrame()
        for test_bin in range(self.cur_data_window_ind, len(self.bins)-1):
            if (self.model.model_name == "LSTM" or self.model.model_name == "LSTM_Forecast"):
                print("Test bin: %d" % test_bin)
            test_window = self.data_windows[test_bin]
            y_pred = self.model.make_prediction(test_window[0]) # window = (X, y, t)
            y_pred_max, y_pred_min = self.model.get_max_min_range(y_pred)
            res = {}
            res['Run'] = self.run
            res['Test Bin'] = test_bin
            res['y_pred'] = y_pred
            res['y_pred_max'] = y_pred_max
            res['y_pred_min'] = y_pred_min
            res["OriginalValue"] = test_window[1]
            res["StandardDeviation"] =  statistics.stdev(test_window[1])
            res["mean"] =  statistics.mean(test_window[1])

            df_overall_run_predict = df_overall_run_predict.append(res, ignore_index=True)

            self.model.add_new_training_data_window(test_window)
            # break
            if(self.model.model_name == "LSTM_Forecast"):
                self.model.finetune_model(prev_data_windows=3)
            else:
                self.model.finetune_model(prev_data_windows=6)
            
        
        self.df_overall_run_predict = df_overall_run_predict
        
        # print(df_overall_run_predict)


    def visualize_prediction_runwise(self, plot_prediction = True, interval=10, xticks=None,  ylim=(1, 7)):
        MAX_TIME_SPANS = 8 * 60 * 60
        df = self.get_time_indexed_df(self.df_run_time_bin)
        # x = df['Date_Time'].tolist()
        x = df.index.values
        y = df[TARGET].tolist()

        x = x[:MAX_TIME_SPANS]
        y = y[:MAX_TIME_SPANS]

        


        # w = int((len(x) // 1000) * 1.5)
        fig = plt.figure(figsize=(20, 6)) # 10 is width, 7 is height
        ax = fig.gca()
        plt.plot(x, y)
        ax.set_ylim(ylim)
        xticks = self.test_bin_time_stamps
        if(xticks):
            xticks = [pd.to_datetime(x) for x in xticks]
            if(self.run == 17):
                xticks = xticks[: 26]
            ax.set_xticks(xticks)
            x_ticks_labels = list(xticks)
            for i in range(len(x_ticks_labels)):
                x_ticks_labels[i] = "Test bin: %d\n%s"  % (i,  xticks[i])
            # print(x_ticks_labels)
            ax.set_xticklabels(x_ticks_labels)
        # else:
        #     xfmt = md.DateFormatter('%d %H:%M')
        #     ax.xaxis.set_major_formatter(xfmt)
        #     ax.xaxis.set_major_locator(md.MinuteLocator(interval=interval))

        # plt.legend(loc="best")
        plt.xticks( rotation=45 )
        plt.xlabel("Drilling Timestamps")
        plt.ylabel("%s values" % TARGET)

        for i in range(len(xticks)):
            plt.axvline(x=pd.to_datetime(xticks[i]), color="r", linestyle="--")
        
        if plot_prediction:
            test_bins = self.df_overall_run_predict['Test Bin'].unique()
            if(self.run == 17):
                test_bins = test_bins[: 26]
            for test_bin in test_bins:
                # print(test_bin)
                res_df = self.df_overall_run_predict[self.df_overall_run_predict['Test Bin'] == test_bin].iloc[0]
                # print(type(res_df), len(res_df))
                # print(res_df)
                y_pred, y_pred_smooth, y_pred_max, y_pred_max_smooth, y_pred_min, y_pred_min_smooth = self.get_parsed_pred_data(res_df)
                df_tmp = self.df_run_time_bin[self.df_run_time_bin['Bins'] == test_bin]
                df_tmp['y_pred'] = y_pred_smooth
                df_tmp['y_pred_max'] = y_pred_max_smooth
                df_tmp['y_pred_min'] = y_pred_min_smooth
                # # print(type(df_tmp))
                # # print(df_tmp.columns)
                # x_org = pd.to_datetime(df_tmp[DT].tolist())
                # print(len(x_org))

                df_tmp = self.get_time_indexed_df(df_tmp)
                x_pred = df_tmp.index.values[:len(y_pred)]
                x_org = df_tmp.index.values
                # if(len(x_org) > 60 * 60):
                #     break

                # print("Test bin: %d df_tmp: %d y_pred: %d" % (test_bin, len(df_tmp), len(y_pred)))
                # print(x_pred)
                # plt.plot(x_org2, y_pred_smooth, "r", label="Pred Avg" )
                plt.plot(x_org, df_tmp['y_pred'].tolist(), "r", label="Pred Avg" )
                plt.plot(x_org, df_tmp['y_pred_max'].tolist(), "r--", label="Pred Max" )
                plt.plot(x_org, df_tmp['y_pred_min'].tolist(), "r--", label="Pred Min" )

        # plt.legend(loc="best")
        title = "Model: %s Run: %d" % (self.model.model_name, self.run)
        plt.title(title)
        plt.show()
    
    def get_parsed_pred_data(self, pred_df):
        # print("get_parsed_pred_data")
        y_pred = pred_df['y_pred']
        # print(len(y_pred))
        y_pred_smooth = get_smooth_data(y_pred, N=10)
        y_pred_max = pred_df['y_pred_max']
        y_pred_max_smooth = get_smooth_data(y_pred_max, N=10)
        y_pred_min = pred_df['y_pred_min']
        y_pred_min_smooth = get_smooth_data(y_pred_min, N=10)
        # y_original = pred_df['OriginalValue']
        # print("method over")

        return y_pred, y_pred_smooth, y_pred_max, y_pred_max_smooth, y_pred_min, y_pred_min_smooth

    def classify_to_bins(self, df, bin_type, bin_size):
        # bin_type: depth or time
        # bin_size: foot, minute
        df = df.copy()
        bin_size = int(bin_size)
        if bin_type == 'depth':    
            bin_col = 'Bit_Depth'
            depth_bins = np.arange(math.floor(df[bin_col].min()), math.ceil(df[bin_col].max()) + bin_size, bin_size).tolist()
            bin_labels = depth_bins[:-1]
            df['Bins'] = pd.cut(df[bin_col], bins=depth_bins, labels=bin_labels)#.value_counts()
        if bin_type == 'time':
            time_bin_labels = []
            bin_id = 0
            while len(time_bin_labels) < len(df):
                tmp = [bin_id] * 60 * bin_size
                time_bin_labels += tmp
                bin_id += 1
            time_bin_labels = time_bin_labels[: len(df)]
            df['Bins'] = time_bin_labels
        return df

    def get_bin_stamps(self, df):
        bins = df['Bins'].unique()
        time_stamps = []
        for bin in bins:
            t = df[df['Bins'] == bin].iloc[0].Date_Time
            time_stamps.append(t)
        return time_stamps

    def get_time_indexed_df(self, df):
        # print(type(df))
        df = df.copy(deep=True)
        # df['Date_Time2'] = df['Date_Time'].copy()
        df['Date_Time'] = pd.to_datetime(df['Date_Time'])
        df = df.set_index(['Date_Time'], inplace=False)
        df = df.reindex(pd.date_range(start=df.index[0], end=df.index[-1], freq='1s'))
        # print(len(df))
        return df