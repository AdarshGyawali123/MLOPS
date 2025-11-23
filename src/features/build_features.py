from src.data.make_dataset import load_and_merge_data
import mlflow
import numpy as np
import pandas as pd
import os


def build_features():

    base_path = "/Users/adarshgyawali/Desktop/MLOps/MLOPS/MLOPS"
    merged_data = load_and_merge_data()
    ## Dropping Rows and Columns
    print("Handling NaN values based on user's specified criteria...")

    # Step 1: Calculate NaN percentages for all columns in the current DataFrame
    missing_percentages_initial = merged_data.isnull().sum() / len(merged_data) * 100

    # Identify columns with more than 50% NaN values to drop entirely
    columns_to_drop_entirely = missing_percentages_initial[missing_percentages_initial > 50].index.tolist()

    if columns_to_drop_entirely:
        print(f"Dropping columns with > 50% NaN values: {columns_to_drop_entirely}")
        merged_data.drop(columns=columns_to_drop_entirely, inplace=True)
        print(f"Current shape of merged_data after column drops: {merged_data.shape}")
    else:
        print("No columns found with > 50% NaN values to drop entirely.")

    # Step 2: For remaining columns, drop rows where NaNs exist if the column has <= 50% NaNs
    # Recalculate NaN percentages after potential column drops
    missing_percentages_after_col_drop = merged_data.isnull().sum() / len(merged_data) * 100

    # Identify columns with <= 50% NaN values for row-wise dropping
    columns_for_row_drop = missing_percentages_after_col_drop[missing_percentages_after_col_drop <= 50].index.tolist()

    initial_rows_for_row_drop = merged_data.shape[0]

    if columns_for_row_drop:
        print(f"Dropping rows based on NaN values in columns with <= 50% NaNs: {columns_for_row_drop}")
        # Ensure we only consider columns that actually have NaNs among the selected ones for the subset argument
        columns_with_nans_in_subset = [col for col in columns_for_row_drop if merged_data[col].isnull().any()]
        if columns_with_nans_in_subset:
            merged_data.dropna(subset=columns_with_nans_in_subset, inplace=True)
        else:
            print("No NaN values found in the selected columns for row dropping.")
        final_rows_after_row_drop = merged_data.shape[0]
        print(f"Original number of rows for this step: {initial_rows_for_row_drop}")
        print(f"Number of rows after dropping: {final_rows_after_row_drop}")
        print(f"Number of rows dropped in this step: {initial_rows_for_row_drop - final_rows_after_row_drop}")
    else:
        print("No columns found with <= 50% NaN values that require row dropping.")
        final_rows_after_row_drop = initial_rows_for_row_drop # No rows dropped if no columns to consider

    print("Data cleaning complete.")
    print(merged_data.head())


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


    ##Checking Colinearity within coloumns and with Feature Columns "Approved Flag"

    correlation_threshold = 0.8
    highly_correlated_features = set()
    features_to_drop = set()

    # Get absolute correlations with the target variable 'Approved_Flag'
    # Convert 'Approved_Flag' to numeric based on user's request: P1=1, P2=2, P3=3, P4=0
    if merged_data['Approved_Flag'].dtype == 'object':
        # Define the mapping as requested by the user
        approved_flag_mapping = {'P1': 1, 'P2': 2, 'P3': 3, 'P4': 0}
        merged_data['Approved_Flag'] = merged_data['Approved_Flag'].map(approved_flag_mapping)

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


    mlflow.set_tracking_uri("file:///Users/adarshgyawali/Desktop/MLOps/MLOPS/MLOPS/mlruns")
    mlflow.set_experiment("VS_Code_credit_fraud_pipeline")

    with mlflow.start_run(run_name="build_features"):
        mlflow.log_param("missing_percentages_initial",missing_percentages_initial)
        mlflow.log_param("columns_to_drop_entirely",columns_to_drop_entirely)
        mlflow.log_param("missing_percentages_after_col_drop",missing_percentages_after_col_drop)

        # Identify Rows with <= 50% NaN values for row-wise dropping
        mlflow.log_param("columns_for_row_drop",columns_for_row_drop)
        mlflow.log_param("initial_rows_for_row_drop",initial_rows_for_row_drop)
        mlflow.log_param("Columns that have NAN values to drop rows",columns_with_nans_in_subset)

        
        output_path = os.path.join(base_path, "data/interim/features.csv")
        merged_data.to_csv(output_path, index=False)    
    return merged_data

    
 
if __name__ == "__main__":
    df = build_features()  # whatever your function name is
    import os

    os.makedirs("data/interim", exist_ok=True)
    df.to_csv("data/interim/features.csv", index=False)
    print("features.csv saved in data/interim/")
