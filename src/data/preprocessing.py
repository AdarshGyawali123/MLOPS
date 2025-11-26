
from src.data.make_dataset import load_and_merge_data
import mlflow
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["MLFLOW_TRACKING_URI"]
os.environ["MLFLOW_TRACKING_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"]
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("VS_Code_credit_fraud_pipeline")


def preprocessing():
    base_path = "/Users/adarshgyawali/Desktop/MLOps/MLOPS/MLOPS"
    merged_data = load_and_merge_data()
    ## Dropping Rows and Columns
    print("Handling NaN values based on user's specified criteria...")

    # Step 1: Calculate NaN percentages for all columns in the current DataFrame
    missing_percentages_initial = merged_data.isnull().sum() / len(merged_data) * 100

    # Identify columns with more than 50% NaN values to drop entirely
    columns_to_drop_entirely = missing_percentages_initial[missing_percentages_initial > 13].index.tolist()

    if columns_to_drop_entirely:
        print(f"Dropping columns with > 13% NaN values: {columns_to_drop_entirely}")
        merged_data.drop(columns=columns_to_drop_entirely, inplace=True)
        print(f"Current shape of merged_data after column drops: {merged_data.shape}")
    else:
        print("No columns found with > 13% NaN values to drop entirely.")

    # Step 2: For remaining columns, drop rows where NaNs exist if the column has <= 50% NaNs
    # Recalculate NaN percentages after potential column drops
    missing_percentages_after_col_drop = merged_data.isnull().sum() / len(merged_data) * 100
    # Show only columns that still contain NaN values (exclude 0% for clarity)
    non_zero_missing_percentages = missing_percentages_after_col_drop[missing_percentages_after_col_drop > 0]
    print("Columns with remaining NaNs (%):")
    print(non_zero_missing_percentages)

    # Identify columns with <= 50% NaN values for row-wise dropping
    columns_for_row_drop = missing_percentages_after_col_drop[missing_percentages_after_col_drop <= 13].index.tolist()

    initial_rows_for_row_drop = merged_data.shape[0]

    columns_with_nans_in_subset = []

    if columns_for_row_drop:
        print(f"Dropping rows based on NaN values in columns with <= 13% NaNs: {columns_for_row_drop}")
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
        print("No columns found with <= 13% NaN values that require row dropping.")
        final_rows_after_row_drop = initial_rows_for_row_drop # No rows dropped if no columns to consider

    print("Data cleaning complete.")
    print(merged_data.head())
    os.makedirs("data/processed", exist_ok=True)
    merged_data.to_csv("data/processed/preprocessing.csv", index=False)


    with mlflow.start_run(run_name="build_features"):
        mlflow.log_param("missing_percentages_initial",missing_percentages_initial)
        mlflow.log_param("columns_to_drop_entirely",columns_to_drop_entirely)
        mlflow.log_param("missing_percentages_after_col_drop",missing_percentages_after_col_drop)

        # Identify Rows with <= 50% NaN values for row-wise dropping
        mlflow.log_param("columns_for_row_drop",columns_for_row_drop)
        mlflow.log_param("initial_rows_for_row_drop",initial_rows_for_row_drop)
        mlflow.log_param("Columns that have NAN values to drop rows",columns_with_nans_in_subset)

    return merged_data


if __name__ == "__main__":
    df = preprocessing()  # whatever your function name is
    import os

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/preprocessing.csv", index=False)
    print("features.csv saved in data/interim/")