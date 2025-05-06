import hea_analysis
import pandas as pd
import joblib
import argparse
import sys



def predict_and_append(model_path, input_csv_path, output_csv_path='predicted_output.csv'):
    """
    Loads a trained Random Forest model, makes predictions on input data,
    and appends predictions as a new column to the original CSV.

    Parameters:
        model_path (str): Path to the .joblib model file.
        input_csv_path (str): Path to the CSV file containing predictor variables.
        output_csv_path (str): Path to save the output CSV with appended predictions.
    """
    # Load the trained model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Load the input data
    try:
        X_new = pd.read_csv(input_csv_path)
        X_full = pd.DataFrame(X_new)
        X_new = X_new[hea_analysis.PREDICTOR_VARS]
        
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        sys.exit(1)

    # Make predictions
    try:
        predictions = model.predict(X_new)
    except Exception as e:
        print(f"Error making predictions: {e}")
        sys.exit(1)

    # Append predictions as a new column
    X_full['prediction'] = predictions

    # Save output to file
    try:
        X_full.to_csv(output_csv_path, index=False)
        print(f"Output with predictions saved to: {output_csv_path}")
    except Exception as e:
        print(f"Error writing output CSV: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Append predictions to input data using a trained Random Forest model.")
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model (.joblib)')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file with predictor variables')
    parser.add_argument('--output', type=str, default='predicted_output.csv', help='Path to save output CSV with predictions')

    args = parser.parse_args()

    predict_and_append(args.model, args.input, args.output)

if __name__ == '__main__':
    main()
