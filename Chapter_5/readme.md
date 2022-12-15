# Vibration Prediction model


## File description
**Utils.py**
* This file contains some common utily methods

**ResultGenerator.py**
* This file contains some code on how to use this library. This file is used to generate necessary results for the vibration prediction model for the Catoosa dataset.

**Vibration_Model.py**
* This is the main library for the vibration prediction (i.e., VibrationML). This contains all the necessary code that needs to be used. Further API documentation is provided below.



## VibrationML


## Terminology
* Feature: Feature is a tuple of values, e.g., (ROP, Torque, Weight on bits). This can contain any number of features. In the future we can incorporate the time_stamp as a part of this touple for future usecase.
* Data Point: It is a tuple of (features, target variable), e.g., ((torque, weight_on_bits, rop), vibration_values).
* Data Window: A data window is basically a list of datapoints which can be 1, 10, 60, 600 etc



## API description
* VibrationML(model_name): Here the model_name is:
    * LR -> For linear regression model
    * RF -> For Random Forrest model
    * LSTM -> For LSTM based RNN model

* add_new_training_data_window(new_data_window)
     * As new data points keep on coming in real time, we can put them inside a temporary list and when there are sufficient data points (e.g., 600) we can put them inside the model for finetuning.

* finetune_model(prev_data_windows=None)
    * This method is supposed to call after adding new data window via add_new_training_data_window() method.
    * if prev_data_windows is None then all the previous datapoints are used for finetuing. But in many cases we may need to finetune the model only on previous few datapoints (i.e., 1 to 6 data windows which may signify last 10 min to 1 hour of drilling).

* make_prediction(X_test):
    * X_test is a list which may contain only one data points, e.g., (ROP, Torque, Weight on bits)
    * So one use case is as drilling keeps on going the groud values keeps on coming and this make_prediction() method can be used to make prediction on the downhole target parameter, i.e., vibration.

* save_model_weights(model_path=None): 
    * If model path is not provided then the weights of the models will be stored in default path. Different file name is useful for future explanability analysis

* load_model_weights(model_path=None):
    *  If model path is not provided then the weights of the models will be loaded from the default path.

* get_max_min_range(y_pred):
    * [Todo] Needs to update the algorithms to improve preformance
    * Currently it considers the Standard deviation of the vibration of the past W=3 data_windows and uses it to make some prediction on the higher end and the lower end range of the prediction.

<br>

## Usage Example
Sudo code is added for now. For sample usage refer to the sample_usage.py file on how to use this library. Please refer to ResultGenerator.py which is used to generate the graphs used to present the performance of these ML models presented in the powerpoint presentation.. **Library code is updated frequence so name of methods and parameters may be different**


\# *Data Pipeline* (Sample.py for details)

data_window = prepareData() # data_window is a list of datapoints, e.g., [(ROP, weight_bits, Torque), vibration_values] <br>
new_test_data = prepareData() # it is a list of data samples, e.g., [(ROP, weight_bits, Torque)]


\# *Model Finetuning* (Sample.py for details)

model = VibrationML("LR") # initializing linear regression model <br>
model.add_new_training_data_window(new_data_window = data_window) <br>
model.finetune_model(prev_data_windows=1) # if there are more data_windows are added then this value can be more than 1 <br>

\# *Model Prediction* (Sample.py for details)

predicted_vibrations = model.make_prediction(X_test = new_test_data) <br>
y_pred_max, y_pred_min = self.model.get_max_min_range(predicted_vibrations) <br>
\# print_and_visualize_prediction(predicted_vibrations, y_pred_max, y_pred_min)


\# *Prediction Analysis* (Sample.py for details)
You need to write your own code to analyse the performance and the visualization of the predicted values which is out of the scope of this library.


<br><br><br>
## Todo

* Need to implement features to store training dataset to files so that it can handle server restart.
* Data interprolation method needs to be implemented in case the downhole sample rate is less than the sample rate of the feature variables.
* Refactor the code to make it more modular
* Add more code comments.





