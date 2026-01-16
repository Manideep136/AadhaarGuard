# Project: AadhaarGuard - Anomaly Detection System
# Author: [Kolluri Manideep]
# Hackathon: UIDAI Data Hackathon 2026
# Description: Detects fraud in Aadhaar updates using Isolation Forest & Statistical Analysis.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import glob
import os

# ==========================================
# STEP 1: LOAD DATA
# ==========================================
print("--- Starting Project AadhaarGuard ---")

# Helper function to find and load CSV files automatically
def load_files(keyword):
    print(f"Loading {keyword} data...")
    # Finding all csv files in the folder that match the keyword
    path = f"**/*{keyword}*.csv"
    found_files = glob.glob(path, recursive=True)
    
    if len(found_files) == 0:
        print(f"Warning: No files found for {keyword}!")
        return pd.DataFrame() # Return empty if nothing found
    
    # Read and combine all files
    data_frames = []
    for file in found_files:
        try:
            df = pd.read_csv(file)
            data_frames.append(df)
        except:
            pass # Skip bad files
            
    return pd.concat(data_frames, ignore_index=True)

# Load the Enrolment and Biometric datasets
enrolment_df = load_files("enrolment")
biometric_df = load_files("biometric")

# ==========================================
# STEP 2: PRE-PROCESSING
# ==========================================
print("Cleaning and preparing data...")

# We need a standard 'total_activity' column for both datasets
if not enrolment_df.empty:
    # Summing up all age groups to get total daily enrolments
    enrolment_df['total_activity'] = enrolment_df['age_0_5'] + enrolment_df['age_5_17'] + enrolment_df['age_18_greater']
    enrolment_df['type'] = 'Enrolment'

if not biometric_df.empty:
    # Summing up biometric updates
    biometric_df['total_activity'] = biometric_df['bio_age_5_17'] + biometric_df['bio_age_17_']
    biometric_df['type'] = 'Biometric'

# Combine everything into one main dataframe
main_df = pd.concat([enrolment_df, biometric_df], ignore_index=True)

# ==========================================
# STEP 3: ANOMALY DETECTION LOGIC
# ==========================================
print("Running Anomaly Detection Algorithms...")

if not main_df.empty:
    # 1. Calculate the 'Normal' behavior for each district
    # We take the average activity of a district to compare against individual pincodes
    main_df['district_avg'] = main_df.groupby(['state', 'district'])['total_activity'].transform('mean')
    
    # 2. Statistical Rule (The 5x Spike Rule)
    # If a center does 5x more work than the district average, it's suspicious
    main_df['ratio'] = main_df['total_activity'] / main_df['district_avg']
    main_df['is_spike'] = (main_df['total_activity'] > 20) & (main_df['ratio'] > 5)

    # 3. Machine Learning Rule (Isolation Forest)
    # This algorithm is great for finding outliers in unsupervised data
    model = IsolationForest(contamination=0.001, random_state=42)
    # We fill NaNs with 0 just in case
    main_df['iso_score'] = model.fit_predict(main_df[['total_activity']].fillna(0))
    main_df['is_outlier'] = main_df['iso_score'] == -1 # -1 means anomaly
    
    # Combine both rules to find the final list of anomalies
    main_df['is_anomaly'] = main_df['is_spike'] | main_df['is_outlier']

    # ==========================================
    # STEP 4: VISUALIZATION
    # ==========================================
    print("Generating Graph...")
    plt.figure(figsize=(12, 7))
    
    # Plot Normal Data (Blue)
    # taking a 10% sample so the plot doesn't crash with too many points
    normal_data = main_df[main_df['is_anomaly'] == False].sample(frac=0.1, random_state=42)
    plt.scatter(normal_data['district_avg'], normal_data['total_activity'], 
                color='blue', alpha=0.3, s=10, label='Normal Transactions')
    
    # Plot Anomalies (Red)
    suspicious_data = main_df[main_df['is_anomaly'] == True]
    plt.scatter(suspicious_data['district_avg'], suspicious_data['total_activity'], 
                color='red', alpha=0.7, s=25, label='Detected Anomalies')

    # Graph Labels
    plt.title('Fraud Detection: Normal vs Suspicious Activity')
    plt.xlabel('District Average (Expected Activity)')
    plt.ylabel('Pincode Daily Count (Actual Activity)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the graph
    plt.savefig('project_graph.png')
    print("Graph saved as 'project_graph.png'")
    
    # ==========================================
    # STEP 5: SAVE RESULTS
    # ==========================================
    print("Saving report to CSV...")
    
    # Filter only the suspicious rows and sort them by size
    final_report = suspicious_data.sort_values(by='total_activity', ascending=False)
    
    # Select only useful columns
    columns_to_save = ['date', 'state', 'district', 'pincode', 'type', 'total_activity', 'district_avg']
    final_report[columns_to_save].to_csv('suspicious_activity_report.csv', index=False)
    
    print(f"Success! Found {len(final_report)} anomalies.")
    print("Please download 'suspicious_activity_report.csv' and 'project_graph.png'")

else:
    print("Error: No data found. Please check if files are uploaded correctly.")