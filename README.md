# House-Price-Prediction-Project
Overview
This project involves building a machine learning model to predict house prices based on various features such as location, size, and amenities. The dataset used for this project is from [source,the Boston Housing dataset]. The main goal is to provide accurate predictions to help potential buyers, real estate agents, and investors make informed decisions.

Features
     1.Data Cleaning and Preprocessing: Handling missing values, feature scaling, and encoding categorical variables.
     2.Exploratory Data Analysis (EDA): Visualizing data distributions, correlations, and identifying key features influencing house prices.
     3.Model Training and Evaluation: Training various regression models and evaluating their performance using metrics like RMSE and MSE.
     4.Model Comparison: Comparing the performance of different models such as Linear Regression, Decision Tree, and Random Forest.
     5.Pipeline Creation: Building a data processing and modeling pipeline using Scikit-Learn to streamline the workflow.
     6.Model Persistence: Saving the trained model using Joblib for future use.

     
Dataset
The dataset contains the following features:

   1.CRIM: Per capita crime rate by town.
   2.ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.
   3.INDUS: Proportion of non-retail business acres per town.
   4.CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
   5.NOX: Nitric oxides concentration (parts per 10 million).
   6.RM: Average number of rooms per dwelling.
   7.AGE: Proportion of owner-occupied units built prior to 1940.
   8.DIS: Weighted distances to five Boston employment centers.
   9.RAD: Index of accessibility to radial highways.
   10.TAX: Full-value property tax rate per $10,000.
   11.PTRATIO: Pupil-teacher ratio by town.
   12.B: Proportion of African Americans by town.
   13.LSTAT: Percentage of lower status of the population.
   14.MEDV: Median value of owner-occupied homes in $1000s (target variable).
   
Getting Started:
    Prerequisites
    Python 3.7+
    Jupyter Notebook
    Scikit-Learn
    Pandas
    NumPy
    Matplotlib
    Seaborn
    Joblib

Results
The Random Forest Regressor model achieved the best performance with a Root Mean Squared Error (RMSE) of approximately 2.97. The baseline model RMSE was 8.41, indicating a significant improvement.


Conclusion
This project demonstrates the entire workflow of a machine learning project, from data preprocessing and exploration to model training and evaluation. The Random Forest Regressor proved to be the most effective model for this dataset. Future improvements could include hyperparameter tuning, using more advanced models, and incorporating additional feature.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
  Scikit-Learn Documentation
  Pandas Documentation
  Boston Housing Dataset
