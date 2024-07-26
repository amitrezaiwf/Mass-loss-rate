import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

def main(args):

    # Load testing data from CSV file
    data = np.loadtxt(args.test_file)
    x_test = data[:, :-1]  # Assuming last column is the target
    y_test = data[:, -1]

    # Load the trained model
    model = load_model(args.model_file)
    
    # Predict on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model on the test data
    loss, mae = model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss}')
    print(f'Test MAE: {mae}')
    
    # Save predictions to .dat file
    predictions = y_pred
    np.savetxt(args.output_file, predictions, header = 'Actual Predicted', comments = '', fmt = '%f')
    print(f"Predictions saved to {args.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test a Dense Neural Network for regression.')
    parser.add_argument('--test_file', type = str, required = True, help = 'Path to the test data file.')
    parser.add_argument('--model_file', type = str, required = True, help = 'Path to the trained model file.')
    parser.add_argument('--output_file', type = str, required = True, help = 'Save the predictions dat file')
    args = parser.parse_args()
    main(args)
