# Credit Card Fraud Detection

This repository contains the complete workflow for a machine learning (ML) project aimed at detecting credit card fraud using transaction data. The project follows an organized pipeline and implements techniques to ensure robustness, reproducibility, and effective model evaluation.

---

## **Project Overview**
Credit card fraud detection is a critical task that requires identifying fraudulent transactions while minimizing false positives to avoid inconveniencing customers. This project applies machine learning to predict fraudulent transactions using both numerical and categorical features from a dataset.

The pipeline includes:
1. Data ingestion and exploration
2. Feature engineering
3. Model training and evaluation
4. Hyperparameter tuning
5. Financial impact analysis

---
## Installation and Setup

1. Clone the repository - [SSH Link](git@github.com:asifsemail/project1.git)
2. Install the required packages by running the following:

    ```bash
    pip install pandas matplotlib seaborn numpy geopandas hvplot plotly
    ```

3. Download the dataset and place it in the project root director - [Credit Card Transactions Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset/data)
4. Run the Jupyter Notebooks `ak_cc_analysis.ipynb`, `age_jbrooks.ipynb`, `KadeCorrelationAnalysis.ipynb`, `Kadefraudforcasting.ipynb`, `KadeCA2.ipynb`, `merchant.ipynb`, and `SimranAnalyzer.ipynb` to execute the analysis and generate visualizations.

---

## Dataset

The dataset consists of 1.29M records and 983 unique credit card numbers over a 1.5-year period. It includes the following columns:

- `trans_date_trans_time`: Transaction timestamp
- `category`: Transaction category
- `state`: Transaction location (state)
- `amt`: Transaction amount
- `gender`: Customer gender
- `cc_num`: Credit card number (unique customer identifier)
- `job`: Customer job title
- `dob`: Customer date of birth (for calculating age)
- Additional fields for customer details and transaction metadata.

---


## **Features**
The project covers the following key functionalities:
- Automated preprocessing with feature engineering (e.g., time-based features, distance calculations, fraud rates).
- Correlation analysis and feature importance analysis to identify redundant or irrelevant features.
- Stratified data splitting to maintain class balance.
- Implementation of a custom metric: **Weighted Cost-Aware Accuracy (WCAA)** to incorporate financial impact into model evaluation.
- Experiment tracking and optimization using XGBoost and hyperparameter tuning (Grid Search and Randomized Search).
- Modular and reusable code structure for reproducibility.

---


## **Pipeline Overview**

1. **Data Ingestion**

	•	Reads the transaction dataset and prepares it for analysis.
	•	Converts time-related columns into proper datetime format.

2. **Exploratory Data Analysis (EDA)**

	•	Explores data for missing values, unique values, and data imbalances.
	•	Visualizes the distribution of target labels (is_fraud).

3. **Feature Engineering**

	•	Derives new features such as:
	•	Age of the user at the time of transaction.
	•	Time-based features: transaction hour, day, month, and year.
	•	Merchant popularity and transaction count per user.
	•	Distance between user and merchant.
	•	Fraud rate per state.

4. **Data Preprocessing**

	•	Handles missing values and removes irrelevant or personally identifiable information (PII).
	•	Encodes categorical columns using Target Encoding.
	•	Scales numerical features for models sensitive to feature scales (e.g., Logistic Regression).

5. **Correlation Analysis**

	•	Computes a correlation matrix for numeric features to identify redundant or multicollinear features.
	•	Reduces dimensionality by dropping highly correlated features.

6. **Feature Importance Analysis**

	•	Uses XGBoost to rank feature importance and iteratively refines the feature set.

7. **Model Training and Evaluation**

	•	Implements models like Logistic Regression, Random Forest, and XGBoost.
	•	Evaluates using:
	•	Standard metrics: Recall, Precision, F1 Score.
	•	Custom metrics: Weighted Cost-Aware Accuracy (WCAA).

8. **Hyperparameter Tuning**

	•	Optimizes XGBoost using:
	•	Grid Search for exhaustive parameter tuning.
	•	Randomized Search for faster exploration of hyperparameters.

9. **Financial Impact Analysis**

	•	Evaluates the financial cost of model errors using confusion matrix metrics.
	•	Calculates the dollar value WCAA to analyze business impact.

---

## **Models**
### XGBoost
- **Baseline Model** Training Recall Score:
- **Feature-Tuned Model** Training Recall Score:
- **Hyper-Tuned Model** Training Recall Score:
- **Final Model**: Testing Recall Score:
### CatBoost
- **Baseline Model** Training Recall Score: .978
- **Feature-Tuned Model** Training Recall Score: .974
- **Hyper-Tuned Model** Training Recall Score: .972
- **Final Model**: Testing Recall Score: .499
### LightGBM
- **Baseline Model** Training Recall Score: .818
- **Feature-Tuned Model** Training Recall Score: .838
- **Hyper-Tuned Model** Training Recall Score: 1
- **Final Model**: Testing Recall Score: .49
### Logistic Regression
- **Baseline Model** Training Recall Score:
- **Feature-Tuned Model** Training Recall Score:
- **Hyper-Tuned Model** Training Recall Score:
- **Final Model**: Testing Recall Score:
### RandomForest
- **Baseline Model** Training Recall Score:
- **Feature-Tuned Model** Training Recall Score:
- **Hyper-Tuned Model** Training Recall Score:
- **Final Model**: Testing Recall Score:
### Keras
- **Baseline Model** Training Recall Score:
- **Feature-Tuned Model** Training Recall Score:
- **Hyper-Tuned Model** Training Recall Score:
- **Final Model**: Testing Recall Score:
### IsolationForest
- **Baseline Model** Training Recall Score:
- **Feature-Tuned Model** Training Recall Score:
- **Hyper-Tuned Model** Training Recall Score:
- **Final Model**: Testing Recall Score:
#### StackingClassifier(RandomForest, LinearSVC, CatBoostClassifier, AdaBoostClassifier, HistGradientBoosting, LogisticRegression)
- **Baseline Model** Training Recall Score: 0.9774123242
- **Feature-Tuned Model** Training Recall Score: 0.9282742223
- **Hyper-Tuned Model** Training Recall Score: N/A
- **Final Model**: Testing Recall Score: 

---

## **Results**

	•	Baseline Model Performance:
    	•	Recall: X.XXX
    	•	WCAA ($): -XXX.XX
	•	Optimized Model Performance:
    	•	Recall: X.XXX
    	•	WCAA ($): -XXX.XX

---

## Presentation

- **Final Presentation - CC Fraud Modeling**: both pptx and pdf version can be referred

---

## Limitations

- **.py file**: The UI demonstration in the Spenderlytics App folder is specific to this project and file.
- **Fraud Forecasting Accuracy**: Further improvement could be made by integrating more advanced machine learning models for better fraud detection accuracy.

---

## Future Improvements

- **Generalized .py package**: Updating the code to take on any CC file and provide the required analysis as output.
- **Source File**: Future improvment can have a source file that can be pulled by doing API request
- **Real-Time Data Integration**: Implementing real-time analytics to provide up-to-date insights.
- **Customizable Dashboards**: Enhancing data visualization with customizable dashboards for better interaction.

---

## Team

This project was conducted by the following team members:

- **Asif Khan** - Project Manager, ML Engg
- **Jason Brooks** - ML Engg
- **Simranpreet Saini** - ML Engg
- **Amit Gaikwad** - ML Engg
- **Kade Thomas** - ML Engg

---
