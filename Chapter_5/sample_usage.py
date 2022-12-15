''' 
# Data Preparation pipeline
In the webapp this portion of data will come from a csv file or database. For the purpose of the demonstration I am only createing and using dummy dataset
 '''

from Vibration_Model import VibrationML
import numpy as np
import pandas as pd


# generating 1000 dummy features for demonstration
def get_dummy_data(N = 1000):
    # for demonstration, lets assume the features are ROP, Torque, Weight_on_bits
    rop = np.random.random(N) * 1000
    torque =  np.random.random(N) * 300
    weight_on_bits = np.random.randint(100, 110, N)
    # time_stamps = pd.Series(pd.date_range('2016-10-10 09:21:12',
    #                             periods = N, freq = 's')).to_list() # These timestamps will come from csv for future usage.
    accel = 4 + np.random.random(N)
    X = [(rop[i], torque[i], weight_on_bits[i]) for i in range(N)]
    # X = [(rop[i], torque[i], weight_on_bits[i], time_stamps[i]) for i in range(N)]
    y = accel
    return X, y
    
X, y = get_dummy_data(N=1000)

data_window = [(X[i], y[i]) for i in range(600)] # taking first 600 datapoints for train data window
test_X = [X[i] for i in range(601, 700)]
test_y = [y[i] for i in range(601, 700)]
print(type(test_y), test_y)


# Model Finetuning
model = VibrationML("RF")
model.add_new_training_data_window(new_data_window = data_window)
model.add_new_training_data_window(new_data_window = data_window)
model.finetune_model(prev_data_windows=2)


# Prediction Pipeline
predicted_vibrations = model.make_prediction(X_test = test_X)
y_pred_max, y_pred_min = model.get_max_min_range(predicted_vibrations)


# Visualization
print("Model's prediction: ")
print(predicted_vibrations)

print("Model's Max prediction: ")
print(y_pred_max)

print("Model's Min prediction: ")
print(y_pred_min)

print("Actual Values: ")
print(test_y)