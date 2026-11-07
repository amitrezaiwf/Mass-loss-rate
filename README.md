### This repository contains an NN model developed for the work presented in "Grid-based exoplanet atmospheric mass-loss predictions via neural networks" [https://doi.org/10.1051/0004-6361/202452379]. 

### Prerequisites

Before running the model, install the required dependencies:

```bash
pip install -r requirements.txt
```


#### plug-in-ver: This version is tailored for integration within a user-defined modelling framework. 

```python
from plug_in_ver.model import Mass_Loss_Model

# Initialise with planet properties
sim = Mass_Loss_Model(mss = 0.85, d0 = 0.08, euv0 = 955., rpl = 2.4, mpl = 8.88)
rate = sim.calculate_mass_loss_rate()
print(f"Predicted Atmospheric Escape Rate: {rate[0]:.3e} g/s")
```

#### command-line-ver: This version allows users to run the model directly from the command line.

### Command Line Interface (CLI)
Navigate to the command line directory and pass the path to your testing datasets, trained Keras models (`.h5` or SavedModel format), and your scikit-learn scaler objects:

```bash
cd command-line-ver
PYTHONWARNINGS="ignore" TF_CPP_MIN_LOG_LEVEL="3" python3 command-line-ver/test.py --test_file command-line-ver/validation_planets.txt --model_file plug_in_ver/3l_model_lambda_k.h5 --scale_file plug_in_ver/scaler_y_3l_model_lambda_k.save --output_file command-line-ver/predictions.dat
```
#### Arguments:
*   `--test_file`: Path to a whitespace/tab-separated file where the last column is the target variable.
*   `--model_file`: The saved TensorFlow/Keras model file.
*   `--scale_file`: The joblib-serialized scaler used to inverse-transform the predictions.
*   `--output_file`: Target output path where the real-space ($10^x$) mass-loss rate predictions are saved.


#### For unittest: 
```bash
PYTHONWARNINGS="ignore" TF_CPP_MIN_LOG_LEVEL="3" python3 tests/test_model.py
```
