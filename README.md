# Garment-worker-productivity-analysis
1. Project Overview
- This project aims to understand and improve garment worker productivity by using data-driven machine learning techniques. We analyze operational and behavioral features to predict performance, classify inefficiencies, detect anomalies, and interpret influential factors using model explainability tools.

2. Objectives
- Understand how factors like incentive, overtime, and idle time affect worker productivity.
- Use machine learning models to estimate performance and flag inefficiencies.
- Identify underperforming workers early to support timely HR actions.
- Compare model performance using metrics like accuracy and RMSE.
- Apply PCA to explore behavior patterns and highlight unusual trends.
- Use SHAP and LIME to make model decisions clear and explainable.
- Prepare clean, well-structured data for reliable model training.
- Turn data insights into practical strategies for factory and HR managers.

3. Tools and Technologies
- Languages: Python 3
- Libraries: pandas, numpy, scikit-learn, xgboost, shap, lime, matplotlib, seaborn
- Platform: Jupyter Notebook / HTML Export
- Dataset: Garments Worker Productivity Dataset from UCI

4. Implementation Summary
i. Data Preprocessing
   - Cleaned column names, handled nulls, parsed date columns.
   - Created new features: day, month, weekday, and inefficiency flag.

ii. Exploratory Visualization
   - Distribution plots, heatmaps, pair plots, and weekday trends to explore data.

iii. Regression Modeling
   - Trained Linear, Ridge, SVR, GBR, Random Forest, XGBoost, and Voting Regressor.
   - Used cross-validation to evaluate models via R², RMSE, MAE.

iv. Hyperparameter Tuning
   - Applied RandomizedSearchCV on Gradient Boosting to find optimal parameters.

v. Explainability with SHAP and LIME
   - Used SHAP summary and dependence plots for global interpretation.
   - LIME explained individual predictions with local interpretability.

vi. Logistic Classification
   - Classified inefficient workers using logistic regression.
   - Evaluated using confusion matrix, classification report, and ROC curve.

vii. PCA Analysis
   - Reduced dimensions to 2D and 3D for clustering visualization.

viii. Anomaly Detection
   - Applied Isolation Forest to flag outlier behavior based on key features.

5. Example Input and Output
Input Example:
- wip = 300, over_time = 1800, idle_time = 20, incentive = 0

Output Example:
- Predicted Productivity: 0.55
- Inefficiency: Yes
- Anomaly: No
- Top Feature: Overtime (via SHAP)

6. Results
- Best Model: Gradient Boosting 
- Metrics: R² ≈ 0.81, RMSE ≈ 0.06
- Logistic Classifier AUC: ~0.75
- SHAP Insights: over_time, idle_time, and wip are key drivers

7. How to Set Up and Run
1. Install dependencies:
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap lime

2. Download dataset:
   Place garments_worker_productivity.csv in your project directory.

3. Run the notebook:
   Use Jupyter Notebook or convert the HTML back to .ipynb using:
    jupyter nbconvert --to notebook AIT664_Group5.html --output=GarmentProductivity.ipynb

4. Execute cells in order.

8. How to Re-run / Recompile
- Rerun all cells in order.
- Update path to dataset if moved.
- Re-run model training blocks after modifying data.

9. Lessons Learned
- Machine learning models helped predict and improve worker productivity effectively.
- Gradient Boosting and Logistic Regression enabled early detection of inefficiencies.
- Incentive, overtime, and idle time were key drivers of productivity.
- PCA revealed useful patterns in worker behavior.
- SHAP and LIME made the models interpretable and trustworthy.
- Visual tools helped communicate insights to non-technical audiences.
- Data-driven insights support smarter workforce planning and decision-making.
