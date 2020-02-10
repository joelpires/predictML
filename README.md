# PredictML

It is a Flask powered web app that classifies the income based on socio-economic information of US citizens and also predicts house prices of their houses based on their conditions.

It is hosted at https://predictML.herokuapp.com. Some models might cause the server to crash (more common to ensemble methods) due to the fact that they might demand a quantity of memory that exceeds the free plan for Heroku servers.

You can fork this and just run `flask run`.

It covers ML that can go from the basic concepts to the most advanced ones. All this can be seen more in detail in the files `census.ipynb` and `houses.ipynb`. The list below will discriminate the main concepts covered in this project:

#### Exploratory Analysis
- **Boxplots**
- **Scatters**
- **Distributions**
- **...**

#### Data Cleaning
- ###### Missing Values
- ###### Non-sense Values
- ###### Duplicates
- ###### Outliers (Boxplots, Tukey Method, etc.)
- ###### Skewness
- ###### Oversampling/Undersampling
    - **SMOTE**
    - **Random**
- ###### ...

#### Feature Transformation
- ###### Label Encoding
- ###### One Hot Encoding
- ###### Standardization
- ###### Normalization
- ###### Binning
- ###### ...

#### Feature Creation
- ###### Deep Feature Synthesis
- ###### ...

#### Feature Selection/Dimensionality Reduction
- ###### Pearson Correlations
- ###### Uncertainty coefficient matrix + Anova (for categorical/numerical correlations)
- ###### PCA
- ###### T-SNE
- ###### Collinearity Study
- ###### Wrapper Methods
- ###### Embedded Methods
- ###### Backward Elimination
- ###### Filter Method
- ###### Forward Selection
- ###### Recursive Elimination
- ###### SelectKBest
- ###### Extra Trees

#### Shuffle and Split Data

#### Classifiers
- ##### Base Learners
    - ###### Ranfom Forest
    - ###### Decision Tree
    - ###### Naive Bayes
    - ###### K-Nearest Neighbour
    - ###### Logistic Regression
    - ###### Support Vector Machines
    - ###### Linear Discriminant Analysis
    - ###### Stochastic Gradient Descent
    - ###### Neural Networks

- ##### Ensemble Learners - with a previous study of the correlation among the predictions of the base learers
    - ###### Voting (Soft + Hard) - different combinations of the base learners
    - ###### Bagging
    - ###### Stacking
    - ###### Blending - different combinations of the base learners
    - ###### Boosting
        - **AdaBoost**
        - **Gradient Boosting**
        - **CATBoost**
        - **XGBoost**

#### Regressors
- ##### Base Learners
    - ###### Ranfom Forest
    - ###### Linear Regression + Ridge
    - ###### Linear Regression + Lasso
    - ###### Support Vector Regressor

- ##### Ensemble Learners - with a previous study of the correlation among the predictions of the base learers
    - ###### Voting (Soft + Hard) - different combinations of the base learners
    - ###### Bagging
    - ###### Stacking
    - ###### Blending - different combinations of the base learners
    - ###### Boosting
        - **AdaBoost Regressor**
        - **GraBoost Regressor**
        - **CATBoost Regressor**
        - **XGBoost Regressor**

#### Parameter Tuning of the Models
- ###### Random Search
- ###### Grid Search
- ###### Prunning Trees

#### Performance Metrics (Charts and Statistics)
- ###### Classification
    - **Accuracies**
    - **Precision**
    - **Recalls**
    - **F-Scores**
    - **Confusion-Matrix**
    - **ROC Curves**
    - **AUCs**
    - **Precision-Recall Curves**
- ###### Regression
    - **MSEs**
    - **MAEs**
    - **RMSEs**
    - **R-Squareds**

#### Overfitting/Underfitting Study:
- ###### Charts with Test Scores Throughout Different Training Sizes
- ###### Charts with Test Scores vs Cross Validation Scores vs Training Scores


#### Statistical Testing of the Results
- ###### 30 Runs
- ###### Ranking the Algorithms
- ###### Friedman Test
- ###### Nemenyi Test

#### Interpretability/Explainability
- ##### Feature Importance
- ##### Permutation Importance
- ##### SHAP Values

#### Making Pipelines and Prepare to Predict on new Data

#### Deployment of the ML Models in an Web App Online
