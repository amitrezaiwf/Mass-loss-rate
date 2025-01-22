import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import argparse

def main(args):

    # Load testing data from CSV file
    data = np.loadtxt(args.test_file)
    grid_pt_test = data[:, :-1]  # Assuming last column is the target
    mlr_true_test = data[:, -1]

    # Load the trained model
    model = load_model(args.model_file)
    
    # Predict on the test set
    mlr_pred = model.predict(grid_pt_test)
    scaler = joblib.load(args.scale_file) 
    mlr_pred_ln10 = scaler.inverse_transform(mlr_pred)
    mlr_pred_org = 10**(mlr_pred_ln10)
    
    # Save predictions to .dat file
    np.savetxt(args.output_file, mlr_pred_org, comments = '', fmt = '%f')
    print(f"Predictions saved to {args.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test a Dense Neural Network for regression.')
    parser.add_argument('--test_file', type = str, required = True, help = 'Path to the test data file.')
    parser.add_argument('--model_file', type = str, required = True, help = 'Provide the trained model.')
    parser.add_argument('--scale_file', type = str, required = True, help = 'Provide the scale file.')
    parser.add_argument('--output_file', type = str, required = True, help = 'Save the predictions dat file')
    args = parser.parse_args()
    main(args)
