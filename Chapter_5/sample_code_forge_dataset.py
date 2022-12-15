from Vibration_Model import VibrationML
import numpy as np
import pandas as pd

def get_forge_data(dataset_file, run, features, target):
    forge_df = pd.read_csv(dataset_file)
    print(len(forge_df))

    run1 = forge_df[forge_df['Run'] == run]
    print("Run 1 length: %d" % len(run1) )

    # filtering out samples where ROP is 0 which signifies no drilling activity.
    filterd_df = run1[run1['ROP'] > 0]

    y = list(filterd_df[target].values)
    X = []
    for index, row in filterd_df.iterrows():
        sample = tuple(row[Features].values)
        X.append(sample)

    # print(type(X[0]), type(y))
    return X, y

def get_data_windows(X, y, window_num, window_size=600):
    start = window_size * window_num
    end = start + window_size
    X_window = X[start: end]
    y_window = y[start: end]
    data_window = [(X_window[i], y_window[i]) for i in range(window_size)]
    return X_window, y_window, data_window


dataset_file = r"C:\Alamin\courses\XGeoML\github\XGeoML\vibration_prediction\dataset\Forge_78B_32_Standard_3_Runs.csv"
run = '433-2700ft'
Features = ["ROP", 'Torque', 'Weight_on_Bit']
Target = 'Accel_CSS-007'
Window_Size = 600

X, y = get_forge_data(dataset_file, run, Features, Target)
print(len(X), len(y))


model = VibrationML("RF")

# Training the model on the first data window
X_window, y_window, data_window = get_data_windows(X, y, window_num = 0)
model.add_new_training_data_window(new_data_window = data_window)
model.finetune_model(prev_data_windows=2)

# Now taking the next 4 data window to test incrementally.
# After testing on new data window. Adding this new data windows for finetuning to make prediction on the next data window
for test_window in range(1, 5):
    X_test_window, y_test_window, data_test_window = get_data_windows(X, y, window_num = test_window)
    # print(get_windows(X, y, window_num=window))
    predictions = model.make_prediction(X_test = X_test_window)
    y_pred_max, y_pred_min = model.get_max_min_range(predictions)
    print("\n\n\n\Test window: %d" % test_window)
    # Visualization
    print("Model's prediction window: %d" % test_window)
    print(predictions)

    print("Model's Max prediction: window: %d" % test_window)
    print(y_pred_max)

    print("Model's Min prediction: window: %d" % test_window)
    print(y_pred_min)

    print("Actual Values: window: %d" % test_window)
    print(y_test_window)
    print ("\n" * 10)
    # break

    # Now finetuning the model on the current test window dataset so that future predictions are better.
    model.add_new_training_data_window(new_data_window = data_test_window)
    model.finetune_model(prev_data_windows=6)