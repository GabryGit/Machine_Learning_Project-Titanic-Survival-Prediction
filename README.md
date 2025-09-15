# Supervised Machine Learning Project: Titanic Survival Prediction


### Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Techniques Used](#techniques-used)
- [Preliminary Phase](#preliminary-phase)
- [Model Validation and Fitting](#model-validation-and-fitting)
- [Model Evaluation](#model-evaluation-and-conclusions)
- [Conclusions and Future Improvements](#conclusions-and-future-improvements)
- [Links and Sources](#links-and-sources)


### Project Overview

This project focuses on building a supervised machine learning model to predict passenger survival on the Titanic. The core objective is to create a model that can accurately classify a passenger as either a "survivor" or "non-survivor" based on their personal attributes.

To solve this classification problem, a Decision Tree Classifier was chosen. This model is well-suited for the task as it works by creating a series of rules from the data, which makes the decision-making process transparent and easy to interpret. This allows us to not only predict outcomes but also to understand which factors most influenced a passenger's chance of survival.

![titanic.jpg](https://github.com/GabryGit/Machine_Learning_Project-Titanic-Survival-Prediction/blob/main/resources/titanic.jpg)

### Data Sources

The analysis uses a dataset containing key passenger information, including Age, Sex, Social Class, and Port of Embarkation. 

[Dataset](https://github.com/GabryGit/Machine_Learning_Project-Titanic-Survival-Prediction/blob/main/titanic_sub-dataset.csv)


### Tools

- Python Libraries:
  - Pandas: Data Analysis
  - Matplotlib: Data Visualisation
  - Scikit-Learn: Machine Learning


### Techniques Used

- ColumnTransfomer
- Pipeline
- K-fold Cross Validation
- ML Model: Decision Tree Classifier
- Model evaluation: Confusion Matrix - Classification Report

  
### Preliminary Phase 

1. **Exploratory data analysis (EDA)**: the main focus was to evaluate how each feature in the dataset correlates with the target variable, passenger survival. Below some of the charts plotted during the analysis.
   
   ![age_ditribution](https://github.com/GabryGit/Machine_Learning_Project-Titanic-Survival-Prediction/blob/main/resources/age_distribution.png)
   ![survivors1](https://github.com/GabryGit/Machine_Learning_Project-Titanic-Survival-Prediction/blob/main/resources/survivors_percentage.png)
   ![survivors2](https://github.com/GabryGit/Machine_Learning_Project-Titanic-Survival-Prediction/blob/main/resources/survivors_percentage-2.png)
   ![survivors3](https://github.com/GabryGit/Machine_Learning_Project-Titanic-Survival-Prediction/blob/main/resources/survivors_percentage-3.png)

   The analysis revealed several strong correlations between passenger attributes and their chances of survival.

    - Gender: Female passengers had a significantly higher probability of surviving.
    - Age: Younger passengers were more likely to survive than older passengers.
    - Social Class: Passengers in higher social classes had a greater chance of survival.


2. **Feature engineering**: this phase involved creating the new features 'Women_1st_class' and 'Man_3d_class'. As identified during the exploratory data analysis, these represent the two extremes in terms of survival probability based on the combination of 'Sex' and 'Pclass' (passenger class).
                        The inclusion of these new columns was expected to simplify the prediction task for the model and improve its overall performance.

3. **Data Preparation**: **ColumnTransformer** was used to automate and streamline the preprocessing of different feature types. The key preprocessing steps were:

      - Handling Missing Values:
        
        - For the Embarked feature, missing values are imputed using the mode, as it's the most frequent and thus most representative value for this categorical variable.
        - For the Age feature, missing values are filled with the median. This is a more robust choice than the mean because it is less sensitive to the upper outliers present in the data, providing a more accurate central value.

      - Encoding Categorical Features:
        - All categorical features are converted into a numerical format that the machine learning model can process. This is achieved through one-hot encoding, which creates new binary columns for each category, preventing the model from misinterpreting them as an ordered sequence.


### Model Validation and Fitting

To ensure the model's optimal performance, we used **GridSearchCV** for hyperparameter tuning. This technique systematically works through multiple combinations of parameter values, cross-validating the model to determine the best-performing set of hyperparameters.
We chose GridSearchCV for a few key reasons:

- Limited Search Space: We only needed to tune one hyperparameter: max_depth.
- Small Parameter Grid: The number of values to test for this hyperparameter was small (just 5), making the search process quick and efficient.
- Efficient Training Time: The model's training time was not long, which allowed for a thorough search of the parameter grid without significant computational cost.

After finding the optimal max_depth value, the Decision Tree Classifier was fitted to the training data. This process, which involves training the model on the data and validating its performance, ensures that it's well-calibrated and ready to make predictions on new, unseen data.

![Pipeline](https://github.com/GabryGit/Machine_Learning_Project-Titanic-Survival-Prediction/blob/main/resources/pipeline.png)

![cross_validation.png](https://github.com/GabryGit/Machine_Learning_Project-Titanic-Survival-Prediction/blob/main/resources/cross_validation.png)

### Model Evaluation

During the Data Exploration phase, we observed that the classes in our target variable (survived vs. not survived) were slightly **imbalanced**, with a 40/60 split.
Given this imbalance, accuracy alone is not a reliable metric for evaluating the model's true performance. Therefore, we need to use additional metrics to get a more complete picture of the supervised model's effectiveness.

- Confusion Matrix Analysis
The model's predictions resulted in the following:
  - True Negatives (TN): 124 - The model correctly predicted that 124 passengers did not survive.
  - True Positives (TP): 55 - The model correctly predicted that 55 passengers survived.
  - False Negatives (FN): 29 - The model incorrectly predicted that 29 passengers would not survive when they actually did.
  - False Positives (FP): 15 - The model incorrectly predicted that 15 passengers would survive when they actually did not.

  ![confusion_matrix.png](https://github.com/GabryGit/Machine_Learning_Project-Titanic-Survival-Prediction/blob/main/resources/confusion_matrix.png)

- Classification Report

  ![classification_report.png](https://github.com/GabryGit/Machine_Learning_Project-Titanic-Survival-Prediction/blob/main/resources/classification_report.png)


### Conclusions and Future Improvements

Overall, the model demonstrates solid performance, with a total **accuracy of 80%** and similarly **high macro/weighted averages for other key metrics**. This indicates that it generally makes **reliable predictions**.

However, a deeper analysis reveals a significant weakness: **the model is much better at predicting "Not Survived" passengers than "Survived" ones**. This is clearly shown by the F1 score, which is 85% for the majority class and only 71% for the minority class. This discrepancy is also visible in the confusion matrix, which shows a high number of False Negatives (29) for a total of only 84 positive instances.

As identified during the Data Exploration phase, the dataset was imbalanced, with fewer passengers in the "Survived" class. This imbalance explains the model's difficulty in correctly predicting the minority class and highlights why accuracy alone can be a misleading metric for evaluating the model's true performance.

To improve the model's performance and address this bias, the following steps are recommended:

- *Balance the Dataset*: The most critical next step is to balance the dataset. Oversampling the "Survived" class is a good option to avoid losing valuable information and patterns, which could happen with undersampling.
- *Tune More Hyperparameters*: Expanding the tuning process to include other hyperparameters of the Decision Tree model, beyond just max_depth, could lead to a more robust model.
- *Explore Other Models*: Consider using a more powerful supervised machine learning algorithm, such as a Random Forest or a Gradient Boosting model, as they often offer superior performance for this type of classification problem.

### Links and Sources
[[Notebook]](https://github.com/GabryGit/Machine_Learning_Project-Titanic-Survival-Prediction/blob/main/ML_Project-Titanic-Survival-Prediction.ipynb)

[[Data Analytics Master-Start2Impact University]](https://www.start2impact.it/master/data-science-analytics/)
