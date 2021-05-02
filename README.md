# Automated Pipeline for Predicting Readings for Objects with Different Characteristics

This pipeline was created in an attempt to automate the prediction of time series. A special feature of pipline is working on multiple rows within a single task. In our case, we predicted the consumption of various resources by buildings.
- Pipeline includes data processing: 
    - removal of anomalies
    - interpolation of missing data.
- Creating a featured space
- Model selection

## How to start
To get started, you will need: 
1) write the get_data function in the data file, which should return all the available time series in this task. You can find an example in our file 
2) write the get_test function

To get started, run the **start.py** file.

To predict the time series, run the **get_prediction.py** file with options you need in get_test function.
