import pandas as pd
import os
import numpy as np



def load_and_merge_data():

    # Path to your local project directory
    base_path = "/Users/adarshgyawali/Desktop/MLOps/MLOPS/MLOPS"

    # Full paths to the Excel files
    file1_path = os.path.join(base_path,"data/raw/case_study1.xlsx")
    file2_path = os.path.join(base_path,"data/raw/case_study2.xlsx")

    # Load both Excel files
    print("Loading datasets from local directory...")
    credit_card_data1 = pd.read_excel(file1_path)
    credit_card_data2 = pd.read_excel(file2_path)

    # # Preview the data
    # print("\nCredit Card Data 1:")
    # print(credit_card_data1.head(5))

    # print("\nCredit Card Data 2:")
    # print(credit_card_data2.head(5))

    # MErge Datasets
    merged_data=pd.merge(credit_card_data1, credit_card_data2, on='PROSPECTID',how ='inner')
    merged_data = merged_data.replace(-99999, np.nan)

    return merged_data


if __name__ == "__main__":
    merged = load_and_merge_data()
    os.makedirs("data/interim", exist_ok=True)
    merged.to_csv("data/interim/merged.csv", index=False)
    print("Merged dataset saved to data/interim/merged.csv")