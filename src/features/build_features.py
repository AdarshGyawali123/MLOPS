from src.data.make_dataset import load_and_merge_data
from src.data.preprocessing import preprocessing
import mlflow
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["MLFLOW_TRACKING_URI"]
os.environ["MLFLOW_TRACKING_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"]


def build_features():
    merged_data=preprocessing()
    base_path = "/Users/adarshgyawali/Desktop/MLOps/MLOPS/MLOPS"

    #Mapping Numerical Values for Education
    print("Applying Ordinal Encoding to 'EDUCATION' column...")
    education_mapping = {
        'SSC': 1,
        '12TH': 2,
        'OTHERS': 2,
        'UNDER GRADUATE': 3,
        'GRADUATE': 3,
        'POST-GRADUATE': 4,
        'PROFESSIONAL': 5
    }
    merged_data['EDUCATION'] = merged_data['EDUCATION'].map(education_mapping)

    # Convert 'Approved_Flag' to numeric based on user's request: P1=1, P2=2, P3=3, P4=0
    if merged_data['Approved_Flag'].dtype == 'object':
        # Define the mapping as requested by the user
        approved_flag_mapping = {'P1': 1, 'P2': 2, 'P3': 3, 'P4': 0}
        merged_data['Approved_Flag'] = merged_data['Approved_Flag'].map(approved_flag_mapping)


    ##Checking Colinearity within coloumns and with Feature Columns "Approved Flag"

    correlation_threshold = 0.8
    highly_correlated_features = set()
    features_to_drop = set()

    # Get absolute correlations with the target variable 'Approved_Flag'

    # Select numerical columns AFTER 'Approved_Flag' has been converted
    numerical_data = merged_data.select_dtypes(include=np.number)

    # Ensure 'Approved_Flag' is in numerical_data and correlation_matrix
    # This check is now mostly redundant if numerical_data is selected after mapping, but good for robustness
    if 'Approved_Flag' not in numerical_data.columns:
        print("Warning: Approved_Flag not found in numerical_data after mapping. Check data types.")

    correlation_matrix = numerical_data.corr()
    approved_flag_correlations = correlation_matrix['Approved_Flag'].abs().sort_values(ascending=False)

    # Iterate through the correlation matrix to find highly correlated pairs
    # Exclude 'Approved_Flag' itself from being considered for dropping against other features based on its own correlation
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]

            # Skip if either column is the target variable
            if col1 == 'Approved_Flag' or col2 == 'Approved_Flag':
                continue

            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                # If both features are highly correlated, keep the one more correlated with 'Approved_Flag'
                # Note: We are using absolute correlation with Approved_Flag to decide which to keep
                if approved_flag_correlations.get(col1, 0) < approved_flag_correlations.get(col2, 0):
                    features_to_drop.add(col1)
                else:
                    features_to_drop.add(col2)

    # Drop the identified features from merged_data
    original_cols_count = merged_data.shape[1]
    merged_data.drop(columns=list(features_to_drop), inplace=True, errors='ignore')

    print(f"Identified {len(features_to_drop)} features to drop based on high correlation (> {correlation_threshold}) with other features and lower correlation with 'Approved_Flag'.")
    print(f"Features dropped: {list(features_to_drop)}")
    print(f"Number of columns before dropping: {original_cols_count}")
    print(f"Number of columns after dropping: {merged_data.shape[1]}")

    # Recalculate correlation matrix for visualization
    numerical_data_after_dropping = merged_data.select_dtypes(include=np.number)
    correlation_matrix_filtered = numerical_data_after_dropping.corr()



    # Identify remaining categorical columns for one-hot encoding
    # Exclude 'EDUCATION' (already ordinal encoded), 'Approved_Flag' (already numerical),
    # and the product inquiry columns (already one-hot encoded)

    remaining_categorical_cols = merged_data.select_dtypes(include='object').columns.tolist()

    # Exclude columns that were already handled by ordinal or one-hot encoding
    # 'EDUCATION' (ordinal)
    # 'MARITALSTATUS' and 'GENDER' will be one-hot encoded now
    # 'last_prod_enq2' and 'first_prod_enq2' were already one-hot encoded in the previous step

    # Filter out 'MARITALSTATUS' and 'GENDER' from `remaining_categorical_cols`
    # and apply one-hot encoding to them

    print(f"Categorical columns identified for one-hot encoding: {remaining_categorical_cols}")

    if remaining_categorical_cols:
        # Log initial state for MLflow
        initial_feature_count_before_one_hot = merged_data.shape[1]
        one_hot_encoded_df = pd.get_dummies(merged_data[remaining_categorical_cols], columns=remaining_categorical_cols, drop_first=False)
        merged_data = pd.concat([merged_data.drop(columns=remaining_categorical_cols), one_hot_encoded_df], axis=1)
        print(f"One-hot encoding applied to {remaining_categorical_cols}.")
        print(f"New shape of merged_data after one-hot encoding: {merged_data.shape}")
    else:
        print("No remaining categorical columns found for one-hot encoding.")

    # print(merged_data.head())


    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("VS_Code_credit_fraud_pipeline")


    os.makedirs("data/interim", exist_ok=True)
    output_path = os.path.join(base_path, "data/interim/features.csv")
    merged_data.to_csv(output_path, index=False)    
    return merged_data

    
 
if __name__ == "__main__":
    df = build_features()  # whatever your function name is
    import os

    os.makedirs("data/interim", exist_ok=True)
    df.to_csv("data/interim/features.csv", index=False)
    print("features.csv saved in data/interim/")
