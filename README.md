# AadhaarGuard: AI-Powered Fraud Detection System

###  The Problem
Manual verification of millions of Aadhaar updates is impossible. Fraudsters exploit this by using automated scripts to generate "Ghost IDs" or process illegal biometric updates at high speeds.

###  The Solution
AadhaarGuard is an automated anomaly detection system that uses Unsupervised Machine Learning (**Isolation Forest**) to flag suspicious patterns in real-time. It separates normal demographic trends from statistical outliers without needing human intervention.

###  Key Findings (from our Analysis)
* **Total Records Analyzed:** ~2 Million
* **Anomalies Detected:** 84,516 (4.2% of traffic)
* **Critical Alert:** Detected a single pincode in South Delhi processing **13,381 updates/day** (100x the district average).
* **Insight:** Biometric Update Fraud is significantly higher than Enrolment Fraud.

### Tech Stack
* **Python:** Core Logic
* **Scikit-Learn:** Isolation Forest Algorithm
* **Pandas:** Data Processing
* **Matplotlib:** Visualization

###  How to Run
1. Upload the dataset CSVs.
2. Run `analysis_script.py`.
3. The system generates a `suspicious_activity_report.csv` and a visualization graph.
