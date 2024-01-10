### From point data to raster data via Gaussian process regression

This directory contains source code for data processing (loading, cleaning, etc.), exploratory data analysis, and Gaussian process regression to convert measurements at specified (x,y) points to raster data, i.e. a mesh of values that interpolate between the point data measurements. 

Most of the heavy-lifting is done in the files *data_ingestion.py*, *explore_data.py*, and *gaussian_process.py*. 

### Project structure
|── README.md
├── data
│   └── data.csv
├── figures
├── logs
│   └── trial_plots
│       ├── mean
|       |-- standard_deviation
├── requirements.txt
├── src
│   ├── data_ingestion.py
│   ├── explore_data.py
│   ├── gaussian_process.py
│   ├── run_final.py
│   └── run_trials.py
└── tests
    ├── empty.csv
    ├── non_empty.csv
    └── test_data_loading.py

### Dependencies
To install the necessary dependencies, type the following on the command line from the GaussianProcess directory:
pip install -r requirements.txt

### To run the code and generate the final output (image files), please type the following on the command line:
1. cd src/
2. python run_final.py

run_final.py goes through 3 steps:
1. Ingest and clean the data stored in /data/data.csv
2. Run Gaussian process regression on one best overall model and create final output 
    measurements interpolated onto a grid of specified resolution.
    -- Measurements are overplotted as red circles with radii corresponding to the measurement value (wider circle =  higher value)
3. Do a preliminary uncertainty analysis
    -- Images show the standard deviation of the interpolated measurement mesh, which illustrates where the predictions have highest and lowest uncertainty
    -- Quantify the 95% credible interval, transformed from log10 space back to regular space. Histograms for this interval width divided by the predicted concentration are shown in Figures ...

### To run a parameter scan:
1. cd src/
2. python run_trials.py

run_trials.py goes through 3 steps:
1. Ingest and clean the data stored in /data/data.csv
2. Explore and visualize the data
    -- Create histograms of measurements, both in log10 and non-log10 space
    -- Create scatter plots of measurements, both in log10 and non-log10 space
3. Run a parameter scan
    -- Test various input parameters and kernels for Gaussian process regression.
    -- Save the marginal log-likelihood and mean squared error for each model in a text file in the /logs/ directory.
    -- Save images of the interpolated values and their standard deviations in the /logs/trial_plots/ directory 


### Unit tests
Unit tests, all for the data ingestion and preprocessing phase, are in the /tests/ directory.
To run the tests from the GaussianProcess directory, type:
pytest tests/*.py

One should see an output like this:
chadbustard@Chads-Air GaussianProcess % pytest tests/test_data_loading.py 
======================================= test session starts =======================================
platform darwin -- Python 3.11.4, pytest-7.4.3, pluggy-1.3.0
rootdir: /Users/chadbustard/Desktop/GaussianProcess
plugins: anyio-3.7.1
collected 7 items                                                                                 

tests/test_data_loading.py .......                                                          [100%]

======================================== 7 passed in 0.38s ========================================


    