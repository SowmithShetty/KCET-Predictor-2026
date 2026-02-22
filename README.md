# KCET-Predictor-2026
The KCET College Predictor 2026 is a machine learning-based web application designed to help Karnataka students estimate their engineering college allotment chances. While most predictors use simple search queries, this tool employs a Random Forest Regressor to forecast 2026 cutoffs based on multi-year trends (2023–2025).

✨ Key Features
Predictive Intelligence: Forecasts the 2026 cutoff rank rather than just showing last year's data.

Interactive Trend Analysis: Dynamic Plotly line charts showing the "Cutoff Journey" from 2023 to 2026.

Comprehensive Filters: Full support for:

Regions: General Karnataka & Hyderabad-Karnataka (HK).

Categories: GM, SC, ST, 1, 2A, 2B, 3A, 3B.

Quotas: Rural and Kannada Medium benefits.

Real-time Results: Instant feedback with "High Chance" or "Borderline" status indicators.

🧠 How It Was Built (The Technical Details)
1. Data Engineering

Source: 17+ raw KEA cutoff PDFs converted into structured CSVs.

Volume: Processed over 150,000+ rows of historical data.

Cleaning: Standardized inconsistent branch names (e.g., mapping "CS Computers" to "Computer Science and Engineering").

2. Machine Learning Pipeline

Algorithm: Random Forest Regressor.

Optimization: Applied Log Transformation to the target variable (Cutoff_Rank) to stabilize predictions across highly skewed rank distributions (1 to 200,000+).

Encoding: Used LabelEncoder to transform categorical data (Colleges, Branches, Regions) into a mathematical format.

3. Frontend & Deployment

UI Framework: Streamlit (Python-based web app).

Visuals: Plotly Express for high-fidelity forecasting graphs.

Deployment: Hosted on Streamlit Community Cloud for global accessibility.

🛠️ Tech Stack
Language: Python 3.10+

ML Libraries: Scikit-Learn, Joblib

Data Science: Pandas, NumPy

Visualization: Plotly

Web: Streamlit

📂 Repository Structure
app.py: The main web application script.

kcet_model_turbo.pkl: The trained AI model.

kcet_ai_ready.csv: The cleaned master dataset.

encoder_*.pkl: Pre-trained encoders for translating user input into AI-readable format.

requirements.txt: List of necessary Python libraries for deployment.
