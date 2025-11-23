from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os



## WSGI Application
app = Flask(__name__)

# @app.route("/")
# def healthcheck():
#     return("Hi You are in localhost")

with open ("./models/model.pkl","rb") as f:
    model = pickle.load(f)



FEATURE_COLUMNS = [
    "PROSPECTID",
    "Total_TL",
    "Tot_Active_TL",
    "Total_TL_opened_L6M",
    "pct_tl_open_L6M",
    "pct_tl_closed_L6M",
    "Tot_TL_closed_L12M",
    "pct_tl_open_L12M",
    "pct_tl_closed_L12M",
    "Tot_Missed_Pmnt",
    "Auto_TL",
    "CC_TL",
    "Home_TL",
    "PL_TL",
    "Unsecured_TL",
    "Other_TL",
    "Age_Oldest_TL",
    "Age_Newest_TL",
    "time_since_recent_payment",
    "num_deliq_6_12mts",
    "max_deliq_12mts",
    "num_times_60p_dpd",
    "num_std_12mts",
    "num_sub",
    "num_sub_6mts",
    "num_sub_12mts",
    "num_dbt",
    "num_dbt_12mts",
    "num_lss",
    "num_lss_6mts",
    "num_lss_12mts",
    "recent_level_of_deliq",
    "CC_enq_L6m",
    "PL_enq_L6m",
    "time_since_recent_enq",
    "enq_L3m",
    "EDUCATION",
    "AGE",
    "NETMONTHLYINCOME",
    "Time_With_Curr_Empr",
    "pct_of_active_TLs_ever",
    "pct_opened_TLs_L6m_of_L12m",
    "pct_currentBal_all_TL",
    "CC_Flag",
    "PL_Flag",
    "pct_PL_enq_L6m_of_L12m",
    "pct_CC_enq_L6m_of_L12m",
    "max_unsec_exposure_inPct",
    "HL_Flag",
    "Credit_Score"
]


# NOTE: APPROVED_FLAG and PROSPECTID are intentionally NOT in FEATURE_COLUMNS.
# If your model was trained including them (not ideal), adjust this list accordingly.

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="No file part in the request.")
    
    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    try:
        # Read Excel into DataFrame
        df = pd.read_excel(file)

        # Optional: show what columns came in (for debugging)
        # print("Uploaded columns:", df.columns.tolist())

        # Check for required columns
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            msg = f"Missing required columns: {', '.join(missing_cols)}"
            return render_template("index.html", error=msg)
        
        education_mapping = {
                'SSC': 1,
                '12TH': 2,
                'OTHERS': 2,
                'UNDER GRADUATE': 3,
                'GRADUATE': 3,
                'POST-GRADUATE': 4,
                'PROFESSIONAL': 5
            }

        if 'EDUCATION' in df.columns:
            # If already numeric, do nothing
            if df['EDUCATION'].dtype == 'object':
                df['EDUCATION'] = df['EDUCATION'].str.upper().map(education_mapping)



        # Extract features in correct order
        X = df[FEATURE_COLUMNS]

        # Run model prediction
        preds = model.predict(X)

        # Attach predictions to dataframe
        # If PROSPECTID exists, keep it; if not, we just show row index
        if "PROSPECTID" in df.columns:
            result_df = df[["PROSPECTID"]].copy()
        else:
            result_df = df.reset_index().rename(columns={"index": "Row"})

        result_df["Prediction"] = preds

        # Convert to a pretty HTML table
        html_table = result_df.to_html(classes="result-table", index=False)

        return render_template("results.html", table=html_table)

    except Exception as e:
        # Print full error for debugging in console
        print("Error while processing file:", e)
        return render_template("index.html", error=f"Error processing file: {e}")



if __name__ == "__main__":
    app.run(debug=True)

