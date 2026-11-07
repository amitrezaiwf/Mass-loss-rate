import os
import sys
import warnings
warnings.filterwarnings("ignore", category = UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#-

import argparse
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
#-

def main(args):
    
    print("==================================================================")
    print("          INITIALIZING COMMAND-LINE BATCH INFERENCE UTILITY       ")
    print("==================================================================\n")
    if not os.path.exists(args.test_file):
        sys.exit(f"Execution Aborted: Data file '{args.test_file}' does not exist.")
    print(f" Reading input files: {args.test_file}")
    
    # Load testing data from CSV file
    data = np.loadtxt(args.test_file)
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis = 0)
    
    num_cols = data.shape[1]

    if num_cols >= 7:
        grid_pt_test = data[:, :6]
        mlr_true_test = data[:, -1]
        has_truth = True
        print(f"test file with groud truth is available. Processing {data.shape[0]} profiles.")
    elif num_cols ==6: 
        grid_pt_test = data
        has_truth = False
        print(f"test file with only input parameters. Processing {data.shape[0]} profiles.")
    else:
        sys.exit(f" Error: Expected input file to have 6 or 7 columns. Found {num_cols} columns.")

    print(f" Compiling trained Neural Network weights: {args.model_file}")
    model = load_model(args.model_file, compile = False)
    mlr_pred = model.predict(grid_pt_test, verbose = 0) # Predict on the test set
    scaler = joblib.load(args.scale_file)
    mlr_pred_ln10 = scaler.inverse_transform(mlr_pred)
    mlr_pred_org = 10**(mlr_pred_ln10)

    # Save predictions to .dat file
    np.savetxt(args.output_file, mlr_pred_org, comments = '', fmt = '%e')
    print(f"Predictions saved to {args.output_file}")
    
    # OPTIONAL: If the file has a 7th colum
    if has_truth:
        print("\n====================================================")
        print(" INSTANT QUANTITATIVE MODEL VALIDATION CHECK")
        print("======================================================")
        true_vals = mlr_true_test.flatten()
        pred_vals = mlr_pred_org.flatten()
        relative_residuals = (np.abs(pred_vals - true_vals) / true_vals) * 100
        for idx in range(len(true_vals)):
            print(f"Profile Row {idx + 1:02d} -> Relative Error: {relative_residuals[idx]:.4f}%")

        print("=======================================================")

        print(f"→ Overall Average Prediction Error: {np.mean(relative_residuals):.4f}%")
        print("======================================================")

    print("\nInference run successfully completed.\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Command-line evaluation pipeline interface for exoplanet mass-loss neural networks.')
    parser.add_argument('--test_file', type = str, required = True, help = 'Path to the test data file.')
    parser.add_argument('--model_file', type = str, required = True, help = 'Provide the trained model.')
    parser.add_argument('--scale_file', type = str, required = True, help = 'Provide the scale file.')
    parser.add_argument('--output_file', type = str, required = True, help = 'Save the predictions dat file')

    args = parser.parse_args()
    main(args)
