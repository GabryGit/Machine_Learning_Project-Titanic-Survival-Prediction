# Supervised Machine Learning Project: Titanic Survival Prediction


### Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Techniques Used](#techniques-used)
- [Preliminary Phase](#preliminary-phase)
- [Analysis Results ](#analysis-results)
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

- ColumnTransfomer (Data Preparation)
- Pipeline
- K-fold Cross Validation
- ML Model: Decision Tree Classifier
- Model evaluation: Confusion Matrix - Classification Report

  
### Preliminary Phase 

1. During the exploratory data analysis (EDA) phase, the main focus was to evaluate how each feature in the dataset correlates with the target variable, passenger survival. Below some of the charts plotted during the analysis
   ![age_ditribution](https://github.com/GabryGit/Machine_Learning_Project-Titanic-Survival-Prediction/blob/main/resources/age_distribution.png)
   ![survivors1](https://github.com/GabryGit/Machine_Learning_Project-Titanic-Survival-Prediction/blob/main/resources/survivors_percentage.png)
   ![survivors2](https://github.com/GabryGit/Machine_Learning_Project-Titanic-Survival-Prediction/blob/main/resources/survivors_percentage-2.png)
   ![survivors3](https://github.com/GabryGit/Machine_Learning_Project-Titanic-Survival-Prediction/blob/main/resources/survivors_percentage-3.png)

### Analysis Results 

Here a print of the EMS running: 

![Python-print](https://github.com/GabryGit/Python_Project-Fitness-application-EMS/blob/main/Sources/python-print.png)


### Links and Sources
[Python script](https://github.com/GabryGit/Python_Project-Fitness-application-EMS/blob/main/Python_Project-Fitness-app-EMS.ipynb)

[[Data Analytics Master-Start2Impact University]](https://www.start2impact.it/master/data-science-analytics/)
