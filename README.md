# PredictML

It is a Flask powered web app that classifies the income based on socio-economic information of US citizens and also predicts house prices of their houses based on their conditions.

It is hosted at https://predictML.herokuapp.com. Some models might cause the server to crash (more common to ensemble methods) due to the fact that they might demand a quantity of memory that exceeds the free plan for Heroku servers.

You can fork this and just run `flask run`.

It covers ML that can go from the basic concepts to the most advanced ones. All this can be seen more in detail in the files `census.ipynb` and `houses.ipynb`. The list below will discriminate the main concepts covered in this project:

### Exploratory Analysis
- **Boxplots**
- **Scatters**
- **Distribution Bars**
- **...**

### Data Cleaning
- **Missing Values**
- **Non-sense Values**
- **Duplicates**
- **Outliers (Boxplots, Tukey Method, etc.)**
- **Skewness**
- **Oversampling/Undersampling**
    - **SMOTE**
    - **Random**
- **...

### Feature Transformation
- **Label Encoding**
- **One Hot Encoding**
- **Standardization**
- **Normalization**
- **Binning**
- **...**

### Feature Creation
- **Deep Feature Synthesis**
- **...**

### Feature Selection/Dimensionality Reduction
- **Pearson Correlations**
- **Uncertainty coefficient matrix + Anova (for categorical/numerical correlations)**
- **PCA**
- **T-SNE**
- **Collinearity Study**
- **Wrapper Methods**
- **Embedded Methods**
- **Backward Elimination**
- **Filter Method**
- **Forward Selection**
- **Recursive Elimination**
- **SelectKBest**
- **Extra Trees**

### Shuffle and Split Data

### Classifiers
- **Base Learners**
    - **Ranfom Forest**
    - **Decision Tree**
    - **Naive Bayes**
    - **K-Nearest Neighbour**
    - **Logistic Regression**
    - **Support Vector Machines**
    - **Linear Discriminant Analysis**
    - **Stochastic Gradient Descent**
    - **Neural Networks**

- **Ensemble Learners** - with a previous study of the correlation among the predictions of the base learers
    - **Voting (Soft + Hard) - different combinations of the base learners**
    - **Bagging**
    - **Stacking**
    - **Blending** - different combinations of the base learners
    - **Boosting**
        - **AdaBoost**
        - **Gradient Boosting**
        - **CATBoost**
        - **XGBoost**

### Regressors
- **Base Learners**
    - **Ranfom Forest**
    - **Linear Regression + Ridge**
    - **Linear Regression + Lasso**
    - **Support Vector Regressor**

- **Ensemble Learners** - with a previous study of the correlation among the predictions of the base learers
    - **Voting (Soft + Hard)** - different combinations of the base learners
    - **Bagging**
    - **Stacking**
    - **Blending** - different combinations of the base learners
    - **Boosting**
        - **AdaBoost Regressor**
        - **GraBoost Regressor**
        - **CATBoost Regressor**
        - **XGBoost Regressor**

### Parameter Tuning of the Models
- **Random Search**
- **Grid Search**
- **Prunning Trees**

### Performance Metrics (Charts and Statistics)
- **Classification**
    - **Accuracies**
    - **Precision**
    - **Recalls**
    - **F-Scores**
    - **Confusion-Matrix**
    - **ROC Curves**
    - **AUCs**
    - **Precision-Recall Curves**
- **Regression**
    - **MSEs**
    - **MAEs**
    - **RMSEs**
    - **R-Squareds**

### Overfitting/Underfitting Study:
- **Charts with Test Scores Throughout Different Training Sizes**
- **Charts with Test Scores vs Cross Validation Scores vs Training Scores**


### Statistical Testing of the Results
- **30 Runs**
- **Ranking the Algorithms**
- **Friedman Test**
- **Nemenyi Test**

### Interpretability/Explainability
- **Feature Importance**
- **Permutation Importance**
- **SHAP Values**

### Making Pipelines and Prepare to Predict on new Data

### Deployment of the ML Models in an Web App Online


# Classify Annual Income of US Citizens Based on their Socio-Economic Information (<=50K or >50K dollars)

```python
"""
October 2019
Author: Joel Pires
"""

import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
import missingno
import scipy.stats as st
import scikitplot as skplt
import warnings
import statsmodels.api as sm
import featuretools as ft
import pickle
import scikit_posthocs as sp
import imgkit
import os
import plotly as py
import cufflinks as cf
import plotly.graph_objs as go
import copy
import dython.nominal as nominal
import itertools
import shap
import joblib

from eli5 import show_weights
from eli5.sklearn import PermutationImportance
from operator import add, truediv
from shutil import copyfile
from plotly.offline import init_notebook_mode,iplot
from pathlib import Path
from collections import Counter
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, KBinsDiscretizer
from sklearn.metrics import *
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import learning_curve, StratifiedShuffleSplit, train_test_split, cross_val_score, cross_validate, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2, RFE, RFECV, SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree._tree import TREE_LEAF
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge, Lasso
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#sometimes the import of imblearn triggers a bug -.-'
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek

#classifiers evaluated
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from mlens.ensemble import BlendEnsemble
from mlxtend.classifier import EnsembleVoteClassifier
from vecstack import stacking
from sklearn.ensemble import BaggingClassifier

%matplotlib inline
warnings.filterwarnings('ignore')
```


```python
#Load Data
df = pd.read_csv('./data/census/census.csv')
```


```python
categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
numerical = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

#reorder dataframe columns for later better tracking of the features once encoded
original_order = df.columns[:-1].tolist()
new_order = ['age'] + numerical[1:] + categorical
total = new_order + ['income']
df = df[total]
```

# **EXPLORATORY ANALYSIS**


```python
# Preview Data
display(df.shape)
display(df.head())
display(df.tail())
display(df.info())
```


    (32561, 15)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education.num</th>
      <th>capital.gain</th>
      <th>capital.loss</th>
      <th>hours.per.week</th>
      <th>workclass</th>
      <th>education</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>native.country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90</td>
      <td>77053</td>
      <td>9</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>?</td>
      <td>HS-grad</td>
      <td>Widowed</td>
      <td>?</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>82</td>
      <td>132870</td>
      <td>9</td>
      <td>0</td>
      <td>4356</td>
      <td>18</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Widowed</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>66</td>
      <td>186061</td>
      <td>10</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>?</td>
      <td>Some-college</td>
      <td>Widowed</td>
      <td>?</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54</td>
      <td>140359</td>
      <td>4</td>
      <td>0</td>
      <td>3900</td>
      <td>40</td>
      <td>Private</td>
      <td>7th-8th</td>
      <td>Divorced</td>
      <td>Machine-op-inspct</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>264663</td>
      <td>10</td>
      <td>0</td>
      <td>3900</td>
      <td>40</td>
      <td>Private</td>
      <td>Some-college</td>
      <td>Separated</td>
      <td>Prof-specialty</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Female</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education.num</th>
      <th>capital.gain</th>
      <th>capital.loss</th>
      <th>hours.per.week</th>
      <th>workclass</th>
      <th>education</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>native.country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32556</th>
      <td>22</td>
      <td>310152</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Private</td>
      <td>Some-college</td>
      <td>Never-married</td>
      <td>Protective-serv</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32557</th>
      <td>27</td>
      <td>257302</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>Private</td>
      <td>Assoc-acdm</td>
      <td>Married-civ-spouse</td>
      <td>Tech-support</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32558</th>
      <td>40</td>
      <td>154374</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>32559</th>
      <td>58</td>
      <td>151910</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Widowed</td>
      <td>Adm-clerical</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32560</th>
      <td>22</td>
      <td>201490</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 32561 entries, 0 to 32560
    Data columns (total 15 columns):
    age               32561 non-null int64
    fnlwgt            32561 non-null int64
    education.num     32561 non-null int64
    capital.gain      32561 non-null int64
    capital.loss      32561 non-null int64
    hours.per.week    32561 non-null int64
    workclass         32561 non-null object
    education         32561 non-null object
    marital.status    32561 non-null object
    occupation        32561 non-null object
    relationship      32561 non-null object
    race              32561 non-null object
    sex               32561 non-null object
    native.country    32561 non-null object
    income            32561 non-null object
    dtypes: int64(6), object(9)
    memory usage: 3.7+ MB



    None



```python
# Value Counts
for col in df.columns:
    display(df[col].value_counts())
```


    36    898
    31    888
    34    886
    23    877
    35    876
         ...
    83      6
    85      3
    88      3
    87      1
    86      1
    Name: age, Length: 73, dtype: int64



    123011    13
    203488    13
    164190    13
    126675    12
    121124    12
              ..
    36376      1
    78567      1
    180407     1
    210869     1
    125489     1
    Name: fnlwgt, Length: 21648, dtype: int64



    9     10501
    10     7291
    13     5355
    14     1723
    11     1382
    7      1175
    12     1067
    6       933
    4       646
    15      576
    5       514
    8       433
    16      413
    3       333
    2       168
    1        51
    Name: education.num, dtype: int64



    0        29849
    15024      347
    7688       284
    7298       246
    99999      159
             ...
    4931         1
    1455         1
    6097         1
    22040        1
    1111         1
    Name: capital.gain, Length: 119, dtype: int64



    0       31042
    1902      202
    1977      168
    1887      159
    1848       51
            ...
    1411        1
    1539        1
    2472        1
    1944        1
    2201        1
    Name: capital.loss, Length: 92, dtype: int64



    40    15217
    50     2819
    45     1824
    60     1475
    35     1297
          ...
    92        1
    94        1
    87        1
    74        1
    82        1
    Name: hours.per.week, Length: 94, dtype: int64



    Private             22696
    Self-emp-not-inc     2541
    Local-gov            2093
    ?                    1836
    State-gov            1298
    Self-emp-inc         1116
    Federal-gov           960
    Without-pay            14
    Never-worked            7
    Name: workclass, dtype: int64



    HS-grad         10501
    Some-college     7291
    Bachelors        5355
    Masters          1723
    Assoc-voc        1382
    11th             1175
    Assoc-acdm       1067
    10th              933
    7th-8th           646
    Prof-school       576
    9th               514
    12th              433
    Doctorate         413
    5th-6th           333
    1st-4th           168
    Preschool          51
    Name: education, dtype: int64



    Married-civ-spouse       14976
    Never-married            10683
    Divorced                  4443
    Separated                 1025
    Widowed                    993
    Married-spouse-absent      418
    Married-AF-spouse           23
    Name: marital.status, dtype: int64



    Prof-specialty       4140
    Craft-repair         4099
    Exec-managerial      4066
    Adm-clerical         3770
    Sales                3650
    Other-service        3295
    Machine-op-inspct    2002
    ?                    1843
    Transport-moving     1597
    Handlers-cleaners    1370
    Farming-fishing       994
    Tech-support          928
    Protective-serv       649
    Priv-house-serv       149
    Armed-Forces            9
    Name: occupation, dtype: int64



    Husband           13193
    Not-in-family      8305
    Own-child          5068
    Unmarried          3446
    Wife               1568
    Other-relative      981
    Name: relationship, dtype: int64



    White                 27816
    Black                  3124
    Asian-Pac-Islander     1039
    Amer-Indian-Eskimo      311
    Other                   271
    Name: race, dtype: int64



    Male      21790
    Female    10771
    Name: sex, dtype: int64



    United-States                 29170
    Mexico                          643
    ?                               583
    Philippines                     198
    Germany                         137
    Canada                          121
    Puerto-Rico                     114
    El-Salvador                     106
    India                           100
    Cuba                             95
    England                          90
    Jamaica                          81
    South                            80
    China                            75
    Italy                            73
    Dominican-Republic               70
    Vietnam                          67
    Guatemala                        64
    Japan                            62
    Poland                           60
    Columbia                         59
    Taiwan                           51
    Haiti                            44
    Iran                             43
    Portugal                         37
    Nicaragua                        34
    Peru                             31
    France                           29
    Greece                           29
    Ecuador                          28
    Ireland                          24
    Hong                             20
    Cambodia                         19
    Trinadad&Tobago                  19
    Laos                             18
    Thailand                         18
    Yugoslavia                       16
    Outlying-US(Guam-USVI-etc)       14
    Honduras                         13
    Hungary                          13
    Scotland                         12
    Holand-Netherlands                1
    Name: native.country, dtype: int64



    <=50K    24720
    >50K      7841
    Name: income, dtype: int64



```python
# Values Distributions
pd.plotting.scatter_matrix(df.head(500), figsize = (15,9), diagonal = 'hist' )
plt.xticks(rotation = 90)
```




    (array([  0.,  25.,  50.,  75., 100., 125.]),
     <a list of 6 Text xticklabel objects>)




![png](outputs/census/output_6_1.png)



```python
#Relation of Each Feature with the target
fig, ((a,b),(c,d),(e,f)) = plt.subplots(3,2,figsize=(15,20))
plt.xticks(rotation=45)
sns.countplot(df['workclass'],hue=df['income'],ax=f)
sns.countplot(df['relationship'],hue=df['income'],ax=b)
sns.countplot(df['marital.status'],hue=df['income'],ax=c)
sns.countplot(df['race'],hue=df['income'],ax=d)
sns.countplot(df['sex'],hue=df['income'],ax=e)
sns.countplot(df['native.country'],hue=df['income'],ax=a)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f840530208>




![png](outputs/census/output_7_1.png)



```python
#Relation of Each Feature with the target
fig, (a,b)= plt.subplots(1,2,figsize=(20,6))
sns.boxplot(y='hours.per.week',x='income',data=df,ax=a)
sns.boxplot(y='age',x='income',data=df,ax=b)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f840ee65f8>




![png](outputs/census/output_8_1.png)



```python
#additional operations to manipulate data and see it through different perspectives

#df.sort_values(['age','fnlwgt'], ascending = False).head(20)
#df.groupby(df.age).count().plot(kind = 'bar')
#df.groupby('age')['fnlwgt'].mean().sort_values(ascending = False).head(10)
#pd.crosstab(df.age, df.fnlwgt).apply(lambda x: x/x.sum(), axis=1)
#df['young_male'] = ((df.fnlwgt == '19302') & (df.age < 30)).map({True: 'young male', False: 'other'})
#display(df['young_male'])
```

# Data Pre-Processing


## Nonsense values
Negative Values, etc


```python
pd.set_option('display.max_columns', 100)
for col in df:
    print(col)
    print (df[col].unique())
    print('\n')

```

    age
    [90 82 66 54 41 34 38 74 68 45 52 32 51 46 57 22 37 29 61 21 33 49 23 59
     60 63 53 44 43 71 48 73 67 40 50 42 39 55 47 31 58 62 36 72 78 83 26 70
     27 35 81 65 25 28 56 69 20 30 24 64 75 19 77 80 18 17 76 79 88 84 85 86
     87]


    fnlwgt
    [ 77053 132870 186061 ...  34066  84661 257302]


    education.num
    [ 9 10  4  6 16 15 13 14  7 12 11  2  3  8  5  1]


    capital.gain
    [    0 99999 41310 34095 27828 25236 25124 22040 20051 18481 15831 15024
     15020 14344 14084 13550 11678 10605 10566 10520  9562  9386  8614  7978
      7896  7688  7443  7430  7298  6849  6767  6723  6514  6497  6418  6360
      6097  5721  5556  5455  5178  5060  5013  4934  4931  4865  4787  4687
      4650  4508  4416  4386  4101  4064  3942  3908  3887  3818  3781  3674
      3471  3464  3456  3432  3418  3411  3325  3273  3137  3103  2993  2977
      2964  2961  2936  2907  2885  2829  2653  2635  2597  2580  2538  2463
      2414  2407  2387  2354  2346  2329  2290  2228  2202  2176  2174  2105
      2062  2050  2036  2009  1848  1831  1797  1639  1506  1471  1455  1424
      1409  1173  1151  1111  1086  1055   991   914   594   401   114]


    capital.loss
    [4356 3900 3770 3683 3004 2824 2754 2603 2559 2547 2489 2472 2467 2457
     2444 2415 2392 2377 2352 2339 2282 2267 2258 2246 2238 2231 2206 2205
     2201 2179 2174 2163 2149 2129 2080 2057 2051 2042 2002 2001 1980 1977
     1974 1944 1902 1887 1876 1848 1844 1825 1816 1762 1755 1741 1740 1735
     1726 1721 1719 1672 1669 1668 1651 1648 1628 1617 1602 1594 1590 1579
     1573 1564 1539 1504 1485 1411 1408 1380 1340 1258 1138 1092  974  880
      810  653  625  419  323  213  155    0]


    hours.per.week
    [40 18 45 20 60 35 55 76 50 42 25 32 90 48 15 70 52 72 39  6 65 12 80 67
     99 30 75 26 36 10 84 38 62 44  8 28 59  5 24 57 34 37 46 56 41 98 43 63
      1 47 68 54  2 16  9  3  4 33 23 22 64 51 19 58 53 96 66 21  7 13 27 11
     14 77 31 78 49 17 85 87 88 73 89 97 94 29 82 86 91 81 92 61 74 95]


    workclass
    ['?' 'Private' 'State-gov' 'Federal-gov' 'Self-emp-not-inc' 'Self-emp-inc'
     'Local-gov' 'Without-pay' 'Never-worked']


    education
    ['HS-grad' 'Some-college' '7th-8th' '10th' 'Doctorate' 'Prof-school'
     'Bachelors' 'Masters' '11th' 'Assoc-acdm' 'Assoc-voc' '1st-4th' '5th-6th'
     '12th' '9th' 'Preschool']


    marital.status
    ['Widowed' 'Divorced' 'Separated' 'Never-married' 'Married-civ-spouse'
     'Married-spouse-absent' 'Married-AF-spouse']


    occupation
    ['?' 'Exec-managerial' 'Machine-op-inspct' 'Prof-specialty'
     'Other-service' 'Adm-clerical' 'Craft-repair' 'Transport-moving'
     'Handlers-cleaners' 'Sales' 'Farming-fishing' 'Tech-support'
     'Protective-serv' 'Armed-Forces' 'Priv-house-serv']


    relationship
    ['Not-in-family' 'Unmarried' 'Own-child' 'Other-relative' 'Husband' 'Wife']


    race
    ['White' 'Black' 'Asian-Pac-Islander' 'Other' 'Amer-Indian-Eskimo']


    sex
    ['Female' 'Male']


    native.country
    ['United-States' '?' 'Mexico' 'Greece' 'Vietnam' 'China' 'Taiwan' 'India'
     'Philippines' 'Trinadad&Tobago' 'Canada' 'South' 'Holand-Netherlands'
     'Puerto-Rico' 'Poland' 'Iran' 'England' 'Germany' 'Italy' 'Japan' 'Hong'
     'Honduras' 'Cuba' 'Ireland' 'Cambodia' 'Peru' 'Nicaragua'
     'Dominican-Republic' 'Haiti' 'El-Salvador' 'Hungary' 'Columbia'
     'Guatemala' 'Jamaica' 'Ecuador' 'France' 'Yugoslavia' 'Scotland'
     'Portugal' 'Laos' 'Thailand' 'Outlying-US(Guam-USVI-etc)']


    income
    ['<=50K' '>50K']





```python
display(df.describe(include = 'all')) #it helps to understand non-sense values
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education.num</th>
      <th>capital.gain</th>
      <th>capital.loss</th>
      <th>hours.per.week</th>
      <th>workclass</th>
      <th>education</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>native.country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>32561.000000</td>
      <td>3.256100e+04</td>
      <td>32561.000000</td>
      <td>32561.000000</td>
      <td>32561.000000</td>
      <td>32561.000000</td>
      <td>32561</td>
      <td>32561</td>
      <td>32561</td>
      <td>32561</td>
      <td>32561</td>
      <td>32561</td>
      <td>32561</td>
      <td>32561</td>
      <td>32561</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9</td>
      <td>16</td>
      <td>7</td>
      <td>15</td>
      <td>6</td>
      <td>5</td>
      <td>2</td>
      <td>42</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22696</td>
      <td>10501</td>
      <td>14976</td>
      <td>4140</td>
      <td>13193</td>
      <td>27816</td>
      <td>21790</td>
      <td>29170</td>
      <td>24720</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38.581647</td>
      <td>1.897784e+05</td>
      <td>10.080679</td>
      <td>1077.648844</td>
      <td>87.303830</td>
      <td>40.437456</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.640433</td>
      <td>1.055500e+05</td>
      <td>2.572720</td>
      <td>7385.292085</td>
      <td>402.960219</td>
      <td>12.347429</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.000000</td>
      <td>1.228500e+04</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>28.000000</td>
      <td>1.178270e+05</td>
      <td>9.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>37.000000</td>
      <td>1.783560e+05</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48.000000</td>
      <td>2.370510e+05</td>
      <td>12.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>45.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>90.000000</td>
      <td>1.484705e+06</td>
      <td>16.000000</td>
      <td>99999.000000</td>
      <td>4356.000000</td>
      <td>99.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



```python
#CHECK FOR NEGATIVE VALUES BY COLORING THEM
def color_negative_red(val):
    color = 'red' if val <= 0 else 'black'
    return 'color: %s' % color

test = df[['age','capital.gain', 'capital.loss', 'education.num', 'fnlwgt']]
colored = test.head().style.applymap(color_negative_red)

display(colored)
```


<style  type="text/css" >
    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow0_col0 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow0_col1 {
            color:  red;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow0_col2 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow0_col3 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow0_col4 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow1_col0 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow1_col1 {
            color:  red;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow1_col2 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow1_col3 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow1_col4 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow2_col0 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow2_col1 {
            color:  red;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow2_col2 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow2_col3 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow2_col4 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow3_col0 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow3_col1 {
            color:  red;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow3_col2 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow3_col3 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow3_col4 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow4_col0 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow4_col1 {
            color:  red;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow4_col2 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow4_col3 {
            color:  black;
        }    #T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow4_col4 {
            color:  black;
        }</style><table id="T_cee5a7b4_4abc_11ea_90b5_b88687537bdd" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >age</th>        <th class="col_heading level0 col1" >capital.gain</th>        <th class="col_heading level0 col2" >capital.loss</th>        <th class="col_heading level0 col3" >education.num</th>        <th class="col_heading level0 col4" >fnlwgt</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddlevel0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow0_col0" class="data row0 col0" >90</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow0_col1" class="data row0 col1" >0</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow0_col2" class="data row0 col2" >4356</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow0_col3" class="data row0 col3" >9</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow0_col4" class="data row0 col4" >77053</td>
            </tr>
            <tr>
                        <th id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddlevel0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow1_col0" class="data row1 col0" >82</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow1_col1" class="data row1 col1" >0</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow1_col2" class="data row1 col2" >4356</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow1_col3" class="data row1 col3" >9</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow1_col4" class="data row1 col4" >132870</td>
            </tr>
            <tr>
                        <th id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddlevel0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow2_col0" class="data row2 col0" >66</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow2_col1" class="data row2 col1" >0</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow2_col2" class="data row2 col2" >4356</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow2_col3" class="data row2 col3" >10</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow2_col4" class="data row2 col4" >186061</td>
            </tr>
            <tr>
                        <th id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddlevel0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow3_col0" class="data row3 col0" >54</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow3_col1" class="data row3 col1" >0</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow3_col2" class="data row3 col2" >3900</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow3_col3" class="data row3 col3" >4</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow3_col4" class="data row3 col4" >140359</td>
            </tr>
            <tr>
                        <th id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddlevel0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow4_col0" class="data row4 col0" >41</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow4_col1" class="data row4 col1" >0</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow4_col2" class="data row4 col2" >3900</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow4_col3" class="data row4 col3" >10</td>
                        <td id="T_cee5a7b4_4abc_11ea_90b5_b88687537bddrow4_col4" class="data row4 col4" >264663</td>
            </tr>
    </tbody></table>


## Missing Values



```python
missingno.matrix(df, figsize=(12, 5))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f840a605c0>




![png](outputs/census/output_16_1.png)



```python
df.isnull().sum()
```




    age               0
    fnlwgt            0
    education.num     0
    capital.gain      0
    capital.loss      0
    hours.per.week    0
    workclass         0
    education         0
    marital.status    0
    occupation        0
    relationship      0
    race              0
    sex               0
    native.country    0
    income            0
    dtype: int64




```python
#convert '?' to NaN
df[df == '?'] = np.nan
"""
columns = df.columns

imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan,  add_indicator=False)
df = pd.DataFrame(imputer.fit_transform(df))

df.columns = columns

display(df.head())
"""


#fill the NaN values with the mode. It could be filled with median, mean, etc
for col in df.columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

#we could fill every missing values with medians of the columns
#df = data.fillna(df.median())

#also, instead of fill, we can remove every row in which 10% of the values are missing
# df = df.loc[df.isnull().mean(axis=1) < 0.1]

"""
#It is possible to use an imputer. and fill with the values of median, mean, etc
#If imputation technique is used, it is a good practice to add an additional binary feature as a missing indicator.
imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan,  add_indicator=True)
imputer = SimpleImputer(strategy='mean')
imputer = SimpleImputer(strategy='median')
imputer = SimpleImputer(strategy='constant')
df = pd.DataFrame(imputer.fit_transform(df))
"""

"""
# It is also possible to run a bivariate imputer (iterative imputer). However, it is needed to do labelencoding first. The code below enables us to run the imputer with a Random Forest estimator
# The Iterative Imputer is developed by Scikit-Learn and models each feature with missing values as a function of other features. It uses that as an estimate for imputation. At each step, a feature is selected as output y and all other features are treated as inputs X. A regressor is then fitted on X and y and used to predict the missing values of y. This is done for each feature and repeated for several imputation rounds.
# The great thing about this method is that it allows you to use an estimator of your choosing. I used a RandomForestRegressor to mimic the behavior of the frequently used missForest in R.
imp = IterativeImputer(RandomForestRegressor(), max_iter=10, random_state=0)
df = pd.DataFrame(imp.fit_transform(df), columns=df.columns)

#If you have sufficient data, then it might be an attractive option to simply delete samples with missing data. However, keep in mind that it could create bias in your data. Perhaps the missing data follows a pattern that you miss out on.

#The Iterative Imputer allows for different estimators to be used. After some testing, I found out that you can even use Catboost as an estimator! Unfortunately, LightGBM and XGBoost do not work since their random state names differ.
"""
```




    '\n# It is also possible to run a bivariate imputer (iterative imputer). However, it is needed to do labelencoding first. The code below enables us to run the imputer with a Random Forest estimator\n# The Iterative Imputer is developed by Scikit-Learn and models each feature with missing values as a function of other features. It uses that as an estimate for imputation. At each step, a feature is selected as output y and all other features are treated as inputs X. A regressor is then fitted on X and y and used to predict the missing values of y. This is done for each feature and repeated for several imputation rounds.\n# The great thing about this method is that it allows you to use an estimator of your choosing. I used a RandomForestRegressor to mimic the behavior of the frequently used missForest in R.\nimp = IterativeImputer(RandomForestRegressor(), max_iter=10, random_state=0)\ndf = pd.DataFrame(imp.fit_transform(df), columns=df.columns)\n\n#If you have sufficient data, then it might be an attractive option to simply delete samples with missing data. However, keep in mind that it could create bias in your data. Perhaps the missing data follows a pattern that you miss out on.\n\n#The Iterative Imputer allows for different estimators to be used. After some testing, I found out that you can even use Catboost as an estimator! Unfortunately, LightGBM and XGBoost do not work since their random state names differ.\n'



(more examples of operations for replacing values if needed in the future)


```python
"""
#Replace specific values if needed
Replace symbol in whole column
df['age'] = df['age'].str.replace('–', '**', regex = True)


#DROP multiple columns
df.drop(['age' , 'fnlwgt'], axis = 1, inplace = True)


#Find text with regex and replace with nothing
 df = df.replace({
     'age':'[A-Za-z]',
     'fnlwgt': '[A-Za-z]',
 },'',regex = True)


#Example of applying a formula to entire column

 def euro(cell):
     cell = cell.strip('€')
     return cell
 df.Wage = df.Wage.apply(euro)

#Insert value in cell depending on values from other cells
 def impute_age(cols):
     age = cols[0]
     Pclass = cols[1]
     if pd.isnull(Age):
         if Pclass == 1:
             return 37
         else:
             return 24
     else:
         return Age

#CHANGING DATA TYPES
#changing values to float
df[['Value','Wage','Age']].apply(pd.to_numeric, errors = 'coerce')
df.column.astype(float)
df.dtypes
"""
```




    "\n#Replace specific values if needed\nReplace symbol in whole column\ndf['age'] = df['age'].str.replace('–', '**', regex = True)\n\n\n#DROP multiple columns\ndf.drop(['age' , 'fnlwgt'], axis = 1, inplace = True)\n\n\n#Find text with regex and replace with nothing\n df = df.replace({\n     'age':'[A-Za-z]', \n     'fnlwgt': '[A-Za-z]',\n },'',regex = True)\n\n\n#Example of applying a formula to entire column\n\n def euro(cell):\n     cell = cell.strip('€')\n     return cell\n df.Wage = df.Wage.apply(euro)\n\n#Insert value in cell depending on values from other cells\n def impute_age(cols):\n     age = cols[0]\n     Pclass = cols[1]\n     if pd.isnull(Age):\n         if Pclass == 1:\n             return 37\n         else:\n             return 24\n     else:\n         return Age\n         \n#CHANGING DATA TYPES\n#changing values to float\ndf[['Value','Wage','Age']].apply(pd.to_numeric, errors = 'coerce')\ndf.column.astype(float)\ndf.dtypes\n"



## Duplicates
We can confidently remove duplicates now, because when we'll do the oversampling we we'll use the Smote method that won't generate more duplicates (in contrast to RandomOverSampling). Also, by doing this before the oversampling, we are guaranteeing that we'll have exactly 50%-50% of y-labels balance later.


```python
print(df.duplicated().sum()) #check if there are duplicates
df.drop_duplicates(keep = 'first', inplace = True) #get rid of them
```

    24


## Feature Transformation/Creation
- Feature Transformation (Modidy existing features) -> Scaling, normalize, standarize, logarithim, ...
- Feature Creation (Add useful features) -> Modify to new, Combine features, Cluster some feature, ...

### Label Encoding


```python
#using labelencoder
label_encoders = [] #this was needed to reverse label encoding later
le = LabelEncoder()
for feature in categorical:
    new_le = copy.deepcopy(le)
    df[feature] = new_le.fit_transform(df[feature])
    label_encoders.append(new_le)


```

## Finding Outliers

We can detect outliers in 3 ways:

- Standard Deviation
- Percentiles (Tukey method)
- Isolation Forest or LocalOutlierFactor (more appropriate for Anomaly/Fraud Detection Problems)

Then, we can handle them by:
 - Remove them
 - Change them to max/min limit

The definition of outlier is quite dubious, but we can defined them as those values that surpasse the limit of 1.5 * IQR.
In this case, either the standard deviation method or Tukey method are valid options. We just need to try and see which gives better results (if it produces better results at all).


```python
#See outliers through boxplots
df.plot(kind = 'box', sharex = False, sharey = False)
plt.show()
```


![png](outputs/census/output_27_0.png)



```python
# Tukey Method

n = 2 #In this case, we considered outliers as rows that have at least two outlied numerical values. The optimal value for this parameter can be later determined though the cross-validation
indexes = []

for col in df.columns[0:14]:
    Q1 = np.percentile(df[col], 25)
    Q3 = np.percentile(df[col],75)
    IQR = Q3 - Q1

    limit = 1.5 * IQR

    list_outliers = df[(df[col] < Q1 - limit) | (df[col] > Q3 + limit )].index # Determine a list of indices of outliers for feature col

    indexes.extend(list_outliers) # append the found outlier indices for col to the list of outlier indices

indexes = Counter(indexes)
multiple_outliers = list( k for k, v in indexes.items() if v > n )

df.drop(multiple_outliers, axis = 0)

df = df.drop(multiple_outliers, axis = 0).reset_index(drop=True)
print(str(len(multiple_outliers)) + " outliers were eliminated")
```

    2499 outliers were eliminated



```python
#You can try with this method to see if it provides better results
"""
#Setting the min/max to outliers using standard deviation
for col in df.columns[0:14]:
    factor = 3 #The optimal value for this parameter can be later determined though the cross-validation
    upper_lim = df[col].mean () + df[col].std () * factor
    lower_lim = df[col].mean () - df[col].std () * factor

    df = df[(df[col] < upper_lim) & (df[col] > lower_lim)]
"""
```




    '\n#Setting the min/max to outliers using standard deviation\nfor col in df.columns[0:14]:\n    factor = 3 #The optimal value for this parameter can be later determined though the cross-validation\n    upper_lim = df[col].mean () + df[col].std () * factor\n    lower_lim = df[col].mean () - df[col].std () * factor\n\n    df = df[(df[col] < upper_lim) & (df[col] > lower_lim)]\n'



## Dealing with imbalanced data


```python
#Check if there are labels imbalance
sns.countplot(x=df['income'],palette='RdBu_r')
df.groupby('income').size()
```




    income
    <=50K    22841
    >50K      7197
    dtype: int64




![png](outputs/census/output_31_1.png)



```python
df['income'] = df['income'].map({'>50K': 1, '<=50K': 0})

Y = df['income']
X = df.drop('income',axis=1)
```

### Oversampling the data **


```python
# Oversampling using smote
smk = SMOTETomek(random_state = 42)
X, Y = smk.fit_sample(X, Y)

"""
# RANDOM Oversample #####
os = RandomOverSampler() #ratio of one feature and another is 50% and 50% respectively
X, Y = os.fit_sample(X, Y)
"""
```




    '\n# RANDOM Oversample #####\nos = RandomOverSampler() #ratio of one feature and another is 50% and 50% respectively\nX, Y = os.fit_sample(X, Y)\n'




```python
#If dowsample was needed...

"""
# RANDOM Undersample #####
os = RandomUnderSampler() #ratio of one feature and another is 50% and 50% respectively
X, Y = os.fit_sample(X, Y)
"""

"""
#Using nearmiss
nm = NearMiss()
X, Y = nm.fit_sample(X, Y)
"""
```




    '\n#Using nearmiss\nnm = NearMiss()\nX, Y = nm.fit_sample(X, Y)\n'




```python
#making sure that everything is good now
df = pd.concat([X, Y], axis=1)
sns.countplot(x=Y,palette='RdBu_r')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f83f619668>




![png](outputs/census/output_36_1.png)


## Transforming Skewed Continuous Features


```python
def analyze_skew():
    #analyze which features are skewed
    fig = plt.figure(figsize = (15,10))
    cols = 3
    rows = math.ceil(float(df[numerical].shape[1] / cols))
    for i, column in enumerate(numerical):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if df.dtypes[column] == np.object:
            df[column].value_counts().plot(kind = 'bar', axes = ax)
        else:
            df[column].hist(axes = ax)
            plt.xticks(rotation = 'vertical')
    plt.subplots_adjust(hspace = 0.7, wspace = 0.2)
    plt.show()

    # from the plots we can see that there are multiple features skewed. However, the following procedure allows us to see in detail how skewed are they.
    skew_feats = df[numerical].skew().sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skew_feats})

    display(skewness)
```


```python
analyze_skew()
```


![png](outputs/census/output_39_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Skew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>capital.gain</th>
      <td>8.866929</td>
    </tr>
    <tr>
      <th>capital.loss</th>
      <td>4.138455</td>
    </tr>
    <tr>
      <th>fnlwgt</th>
      <td>1.277222</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0.321511</td>
    </tr>
    <tr>
      <th>hours.per.week</th>
      <td>0.109324</td>
    </tr>
    <tr>
      <th>education.num</th>
      <td>-0.195546</td>
    </tr>
  </tbody>
</table>
</div>



```python
#Let's reduce the skew of fnlwgt, capital.gain, capital.loss
skewed = ['fnlwgt', 'capital.gain', 'capital.loss']
features_log_transformed = pd.DataFrame(data=df)
features_log_transformed[skewed] = df[skewed].apply(lambda x: np.log(x + 1)) #it can be other function like polynomial, but generally the log funtion is suitable
```


```python
#check again if is everything ok
analyze_skew()
```


![png](outputs/census/output_41_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Skew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>capital.loss</th>
      <td>3.892085</td>
    </tr>
    <tr>
      <th>capital.gain</th>
      <td>2.407371</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0.321511</td>
    </tr>
    <tr>
      <th>hours.per.week</th>
      <td>0.109324</td>
    </tr>
    <tr>
      <th>education.num</th>
      <td>-0.195546</td>
    </tr>
    <tr>
      <th>fnlwgt</th>
      <td>-0.905550</td>
    </tr>
  </tbody>
</table>
</div>


### Binning
Binning continuous variables prevents overfitting which is a common problem for tree based models like decision trees and random forest.

Let's binning the age and see later if the results improved or not.

Binning with fixed-width intervals (good for uniform distributions). For skewed distributions (not normal), we can use binning adaptive (quantile based).

TO-DO: analyze if binning really improves the model. Include experimenting binning after and before the remove of outliers. Theoretically, if it is a linear mode, and data has a lot of "outliers" binning probability is better. If we have a tree model, then, outlier and binning will make too much difference.


```python
#Binner adaptive
binner = KBinsDiscretizer(encode='ordinal')
binner.fit(X[['age']])
X['age'] = binner.transform(X[['age']])

"""
#Using fixed-width intervals
age_labels = ['infant','child','teenager','young_adult','adult','old', 'very_old']

ranges = [0,5,12,18,35,60,81,100]

df['age'] = pd.cut(df.age, ranges, labels = age_labels)

"""
#age now becomes a category
categorical = ['age', 'workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
numerical = ['fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
```

# Feature Selection
- Feature Selection/Reduction (Remove useless features) -> See feature importance, correlations, Dimensionality reduction,

As we only have 14 features, we are not pressured to make a feature selection/reduction in order to increase drastically the computing time of the algorithms. So, for now, we are going to investigate if there are features extremely correlated to each other. After tuning and choosing the best model, we are revisiting feature selection methods just in case we face overfitting or to see if we could achieve the same results with the chosen model but with fewer features.

## Correlation

Since we have a mix of numerical and categorical variables, we are going to analyze the correlation between them independently and then mixed.

- Numerical & Numerical: Pearson correlation is a good one to use, although there are others.
- Categorical & Categorical: We will make use Chi-squared and uncertainty correlation methods through a library called dython.
- Numerical & Categorical: We can use point biserial correlation (only if categorical variable is binary type), or ANOVA test.


```python
#Simple regression line of various variables in relation to  one other
sns.pairplot(X, x_vars = ['capital.loss', 'hours.per.week', 'education.num'], y_vars = 'age', size = 7, aspect = 0.7, kind = 'reg')
```




    <seaborn.axisgrid.PairGrid at 0x1f841db5d68>




![png](outputs/census/output_46_1.png)



```python
# Overview
sns.pairplot(data = X)
```




    <seaborn.axisgrid.PairGrid at 0x1f841965080>




![png](outputs/census/output_47_1.png)


**Correlations between numerical features**


```python
data = X[X.columns.intersection(numerical)]

plt.subplots(figsize=(20,7))
sns.heatmap(data.corr(method = 'pearson'),annot=True,cmap='coolwarm') # the method can also be 'spearman' or kendall'

#to see the correlation between just two variables
#df['age'].corr(df['capital.gain'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f84a410da0>




![png](outputs/census/output_49_1.png)


**Correlations between categorical features**


```python
data = X[X.columns.intersection(categorical)]

cols = data.columns
clen = cols.size

pairings = list(itertools.product(data.columns, repeat=2))
theils_mat = np.reshape([nominal.theils_u(data[p[1]],data[p[0]]) for p in pairings],(clen,clen))
final = pd.DataFrame(theils_mat, index=cols, columns=cols)

fig, ax = plt.subplots(1,1, figsize=(14,10))
sns.heatmap(final,0,1,ax=ax,annot=True,fmt="0.2f").set_title("Uncertainty coefficient matrix")
plt.show()
```


![png](outputs/census/output_51_0.png)


**Correlations between categorical and numerical features**


```python
for num_feature in numerical:
    for cat_feature in categorical:
        args_list = []
        for unique in X[cat_feature].unique():
            args_list.append(X[num_feature][X[cat_feature] == unique])

        f_val, p_val = st.f_oneway(*args_list) # Calculate f statistics and p value
        print('Anova Result between ' + num_feature, ' & '+ cat_feature, ':' , f_val, p_val)

```

    Anova Result between fnlwgt  & age : 46.5268912453089 4.4569777829507625e-39
    Anova Result between fnlwgt  & workclass : 17.225633165330635 6.288135071856597e-23
    Anova Result between fnlwgt  & education : 2.0441905248254884 0.009762901624964735
    Anova Result between fnlwgt  & marital.status : 12.765005429527205 1.863796006189575e-14
    Anova Result between fnlwgt  & occupation : 4.886910693344456 1.238304197319541e-08
    Anova Result between fnlwgt  & relationship : 4.632298851532457 0.000315172576659874
    Anova Result between fnlwgt  & race : 124.46697285232068 8.419234664072455e-106
    Anova Result between fnlwgt  & sex : 21.376514181262525 3.7850098384784932e-06
    Anova Result between fnlwgt  & native.country : 9.016744621966946 6.041379543094484e-53
    Anova Result between education.num  & age : 375.3390218104468 4.008967e-318
    Anova Result between education.num  & workclass : 182.34501783540682 2.926460563414695e-267
    Anova Result between education.num  & education : 5874.925074869358 0.0
    Anova Result between education.num  & marital.status : 207.67100057268635 4.7265446935979185e-262
    Anova Result between education.num  & occupation : 605.773266243161 0.0
    Anova Result between education.num  & relationship : 246.32120236666677 3.117505747544441e-260
    Anova Result between education.num  & race : 85.96173466306159 7.523971631740549e-73
    Anova Result between education.num  & sex : 24.26929527750691 8.408547862045385e-07
    Anova Result between education.num  & native.country : 15.214672870622104 6.5531373835088584e-102
    Anova Result between capital.gain  & age : 97.54627256174153 9.048233185410699e-83
    Anova Result between capital.gain  & workclass : 26.659564001805485 9.497051132020881e-37
    Anova Result between capital.gain  & education : 57.25662990299335 1.5772001139176394e-171
    Anova Result between capital.gain  & marital.status : 65.30649867210741 3.9069285725268614e-81
    Anova Result between capital.gain  & occupation : 22.491427163876043 1.511115096742876e-54
    Anova Result between capital.gain  & relationship : 63.835305644122805 1.3702987209190776e-66
    Anova Result between capital.gain  & race : 14.388196809574213 9.649791689017093e-12
    Anova Result between capital.gain  & sex : 42.2774531570832 8.0097340213292e-11
    Anova Result between capital.gain  & native.country : 1.849066154215362 0.0008742708362504769
    Anova Result between capital.loss  & age : 66.65056129376039 2.62124407780919e-56
    Anova Result between capital.loss  & workclass : 7.736879107133807 2.2206018659137144e-09
    Anova Result between capital.loss  & education : 25.535166947189456 5.751833981133012e-72
    Anova Result between capital.loss  & marital.status : 70.34549955060385 1.425499427099666e-87
    Anova Result between capital.loss  & occupation : 13.338079870080929 4.410942393103499e-30
    Anova Result between capital.loss  & relationship : 69.97708008709682 3.8088254101184346e-73
    Anova Result between capital.loss  & race : 27.964937122591284 3.1392641188756056e-23
    Anova Result between capital.loss  & sex : 66.70885194981354 3.2351620235406776e-16
    Anova Result between capital.loss  & native.country : 2.2109975580088883 1.6462553536982556e-05
    Anova Result between hours.per.week  & age : 759.6640175162116 0.0
    Anova Result between hours.per.week  & workclass : 103.93195315890651 1.7929268693146216e-151
    Anova Result between hours.per.week  & education : 136.6072666563404 0.0
    Anova Result between hours.per.week  & marital.status : 560.932605751092 0.0
    Anova Result between hours.per.week  & occupation : 182.48938946404172 0.0
    Anova Result between hours.per.week  & relationship : 958.1440531263778 0.0
    Anova Result between hours.per.week  & race : 30.19742409346436 3.943048355869619e-25
    Anova Result between hours.per.week  & sex : 2249.3140628997767 0.0
    Anova Result between hours.per.week  & native.country : 0.9600660300833455 0.5422773966387063


**As the p-values are less than 0.05, there are statistical differences between the means of the features (except for hours.per.week  & native.country). Also, as we could perceive from the heatmaps, there are no highly correlated features. Down bellow are the procedures to drop the highest correlated features, even though it's not needed in this case or it might even result in worse results. Let's then proceed with PCA and Backward Elimination methods anyway**

## PCA


```python
#Scalling first a copy of the data
X_copy = X
col_names = X_copy.columns
features = X_copy[col_names]

scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)

X_copy[col_names] = features

n_comp = len(X_copy.columns)

pca = PCA(n_components=n_comp, svd_solver='full', random_state=1001)
X_pca = pca.fit_transform(X_copy)

print('Variance contributions of each feature:')
for j in range(n_comp):
    print(pca.explained_variance_ratio_[j])
```

    Variance contributions of each feature:
    0.15333401147860812
    0.09017507190925453
    0.08077640443554379
    0.07919458379640437
    0.07488899723544273
    0.0720853390020408
    0.07012768352266005
    0.06833299091935036
    0.0643652090787114
    0.0639435579889216
    0.06009493042000803
    0.049936405472590574
    0.045804112412883596
    0.026940702327580116


As we can see, every feature has a meaningful contribution to variance. So there's no need to use PCA-componentes. Anyway, we can render the graph to see how the pca-components data look like.


```python
colors = ['blue', 'red']
plt.figure(1, figsize=(10, 10))

for color, i, target_name in zip(colors, [0, 1], np.unique(Y)):
    plt.scatter(X_pca[Y == i, 0], X_pca[Y == i, 1], color=color, s=1,
                alpha=.8, label=target_name, marker='.')
plt.legend(loc='best', shadow=False, scatterpoints=3)
plt.title(
        "Scatter plot of the training data projected on the 1st "
        "and 2nd principal components")
plt.xlabel("Principal axis 1 - Explains %.1f %% of the variance" % (
        pca.explained_variance_ratio_[0] * 100.0))
plt.ylabel("Principal axis 2 - Explains %.1f %% of the variance" % (
        pca.explained_variance_ratio_[1] * 100.0))



plt.show()
```


![png](outputs/census/output_58_0.png)



```python
#Making the feature reduction if it was actually needed
#X = pd.DataFrame(data = pca_train, columns = ['pca-1', 'pca-2'])
```

## BACKWARD ELIMINATION (wrapper method 1)

- 1) We select a significance level (SL) to stay in the model
- 2) Fit the full model with all possible predictors
- 3) Loop (while p-value < SL)
    - 4) Consider the predictor with highest p-value.
    - 5) Remove the predictor
    - 6) Fit model without this variable

So basically we are feeding all the possible features to the model and then proceed to iteratively remove
the worst performing features one by one. The metric to evaluate feature performance is pvalue above 0.05 (to keep the feature).


```python
pmax = 1
SL = 0.05 # the smaller the SL, the more features will be removed
cols = list(X.columns)
while (len(X.columns) > 0):

    p_values = []

    Xtemp = X[cols]
    Xtemp = sm.add_constant(Xtemp)
    model = sm.OLS(Y,Xtemp).fit()

    p = pd.Series(model.pvalues.values[1:],index = cols)
    p_max = max(p)
    feature_pmax = p.idxmax()

    if(p_max > SL):
        cols.remove(feature_pmax)
    else:
        break

print('A total of {} features selected: {}'.format(len(cols), cols))

#to apply feature reduction
#X = X[cols]
```

    A total of 14 features selected: ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week', 'workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']


Backward elimination actually tells us that we can eliminate the feature native.country. However, as we have just a few features, we don't have the urge to sacrifice a little bit of the accuracy of the model due to performing speed. In the latter parts of this notebook there are shown more advanced feature selection techniques just in case will be needed in the future.

# Base Line Model
It is useful to have a reference if our models are doing realy good or not


```python
l=pd.DataFrame(Y)
l['baseline'] = 0
k = pd.DataFrame(confusion_matrix(Y,l['baseline']))
print(k)
print("Accuracy: " + str(accuracy_score(Y, l['baseline'])))

"""
#If we just want to perform OLS regression to find out interesting stats:
X1 = sm.add_constant(X)
model = sm.OLS(Y, X1).fit()
display(model.summary())
print(model.pvalues)
"""
```

           0  1
    0  20756  0
    1  20756  0
    Accuracy: 0.5





    '\n#If we just want to perform OLS regression to find out interesting stats:\nX1 = sm.add_constant(X)\nmodel = sm.OLS(Y, X1).fit()\ndisplay(model.summary())\nprint(model.pvalues)\n'




```python
#Saving data before OneHotEncoding to assess future importance later
X_not_encoded = copy.deepcopy(X)
X_not_encoded[X.columns] = StandardScaler().fit_transform(X) #not encoded yet, but scaled!
```

# Constructing the Pipeline | One Hot Encoding | Standardization

After analyzing the data and all the feature engineering possibilities, methods and what results from them, it is important to build an ML pipeline, i.e, a structure that encodes the sequence of all the transformations needed to each feature.

This is particularly usefull for making predictions on new data. We want to transform new prediction data futurely to the standard with which our models were trained.

We couldn't do at first because we had to know first if it would be needed to eliminate outliers or not, it it would be needed to make reduce dimensionality, etc. Those operations can change the number of features left or created when we encode posteriorly.

So, our pipeline will only cover operations that can change the value and number of final features, and not other feature engineering aspects like class imbalance, skewness, outliers, etc.

- 1) We'll revert all the feature transformations absolutely needed to deal with class imbalance, skewness, outliers, pca, etc. For example: to check if a pca was needed, we had to standardize the values; to eliminate outliers, we had to encode the categorical variables before...So, in the end of this, we are able to fit the pipeline with all the data cleaned and with the original structure.
      - Revert scaler -> binning -> label encoding
- 2) Construct and fit the pipeline with:
      - imputer (we want our model to be able to predict new data with missing fields) -> binning -> hot encoding -> scaler -> pca

We don't need to aggregate a preferred classifier right now to the pipeline because we will still assess and tune multiple models.


```python
#inverse scaling
X = pd.DataFrame(scaler.inverse_transform(X))
X.columns = df.columns[:-1]

#inverse binning of 'Age'.
X['age'] = binner.inverse_transform(X[['age']])

#meanwhile, 'Age' becomes numerical again
#inverse label encoding
categories = copy.deepcopy(categorical) #removing age
categories = categories[1:]
for index, feature in enumerate(categories):
    X[feature] = label_encoders[index].inverse_transform(X[feature].astype(int))

bin_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan,  add_indicator=False)),
    ('binning', KBinsDiscretizer(encode='ordinal')),
    ('onehot', OneHotEncoder(categories="auto", handle_unknown='ignore')),
    ('scaler', StandardScaler(with_mean=False))
])

numeric_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan,  add_indicator=False)),
    ('scaler', StandardScaler())
    #('pca', pca()) - in our case, reduction of features was not needed
])

categorical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan,  add_indicator=False)),
    ('onehot', OneHotEncoder(categories="auto", handle_unknown='ignore')),
    ('scaler', StandardScaler(with_mean=False))
    #('pca', pca()) - in our case, reduction of features was not needed
])


preprocessor = ColumnTransformer(transformers = [
        ('bin', bin_transformer, ['age']),
        ('num', numeric_transformer, numerical),
        ('cat', categorical_transformer, categories)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])


temp = pipeline.fit_transform(X)
columns = pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps['onehot'].get_feature_names(['age']).tolist() + numerical + pipeline.named_steps['preprocessor'].transformers_[2][1].named_steps['onehot'].get_feature_names(categories).tolist()
X = pd.DataFrame(temp.toarray(), columns=columns)

#Save the pipeline for later loading and make new predictions
joblib.dump(pipeline, './models/census/preprocessing_pipeline.pkl')

```




['./models/census/preprocessing_pipeline.pkl']




```python
# Here's code if we want to one hot encode just the categorical features without pipeline nd merge them with the numerical data for further standardization!
"""
encoder = OneHotEncoder(categories="auto")
encoded_feat = pd.DataFrame(encoder.fit_transform(df[categorical]).toarray(), columns=encoder.get_feature_names(categorical))

#concatenate without adding null values -.-''
l1=encoded_feat.values.tolist()
l2=df[numerical].values.tolist()
for i in range(len(l1)):
    l1[i].extend(l2[i])

X=pd.DataFrame(l1,columns=encoded_feat.columns.tolist()+df[numerical].columns.tolist())
"""
```




    '\nencoder = OneHotEncoder(categories="auto")\nencoded_feat = pd.DataFrame(encoder.fit_transform(df[categorical]).toarray(), columns=encoder.get_feature_names(categorical))\n\n#concatenate without adding null values -.-\'\'\nl1=encoded_feat.values.tolist()\nl2=df[numerical].values.tolist()\nfor i in range(len(l1)):\n    l1[i].extend(l2[i])\n\nX=pd.DataFrame(l1,columns=encoded_feat.columns.tolist()+df[numerical].columns.tolist())\n'



StandardScaler removes the mean and scales the data to unit variance. However, the outliers have an influence when computing the empirical mean and standard deviation which shrink the range of the feature values as shown in the left figure below. Note in particular that because the outliers on each feature have different magnitudes, the spread of the transformed data on each feature is very different: most of the data lie in the [-2, 4] range for the transformed median income feature while the same data is squeezed in the smaller [-0.2, 0.2] range for the transformed number of households.

StandardScaler therefore cannot guarantee balanced feature scales in the presence of outliers.

MinMaxScaler rescales the data set such that all feature values are in the range [0, 1] as shown in the right panel below. However, this scaling compress all inliers in the narrow range [0, 0.005] for the transformed number of households.

[Source of the Theory](https://stackoverflow.com/questions/51237635/difference-between-standard-scaler-and-minmaxscaler)


```python
# Here's code if we want to standardize without pipeline.
#Standard
"""
col_names = X.columns

features = X[col_names]

scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)

X[col_names] = features
"""
```




    '\ncol_names = X.columns\n\nfeatures = X[col_names]\n\nscaler = StandardScaler().fit(features.values)\nfeatures = scaler.transform(features.values)\n\nX[col_names] = features\n'




```python
#We can also try the minimax and see if it results in better performances
"""
#Minimax
col_names = X.columns

features = X[col_names]

scaler = MinMaxScaler().fit(features.values)
features = scaler.transform(features.values)

X[col_names] = features
"""
```




    '\n#Minimax\ncol_names = X.columns\n\nfeatures = X[col_names]\n\nscaler = MinMaxScaler().fit(features.values)\nfeatures = scaler.transform(features.values)\n\nX[col_names] = features\n'



# Split Data

### Shuffle before split


```python
l1=X.values.tolist()
l2=pd.DataFrame(Y).values.tolist()
for i in range(len(l1)):
    l1[i].extend(l2[i])

new_df=pd.DataFrame(l1,columns=X.columns.tolist()+pd.DataFrame(Y).columns.tolist())

new_df = shuffle(new_df, random_state=42)
```


```python
testSize = 0.3 #we can try with different test_sizes

#dividing features that were not hotencoded for frther analysis of the feature importance for each classifier
X_original_train, X_original_test, y_original_train, y_original_test = train_test_split(X_not_encoded, Y, test_size = 0.2)

train,test = train_test_split(new_df,test_size=testSize)

y_train = train['income']
X_train = train.drop('income',axis=1)
y_test = test['income']
X_test = test.drop('income',axis=1)
```

# Building Classifiers and Finding their Best Parameters
Just tuning the crutial parameters, not all of them.
I intercalate between randomizedSearch and GridSearch to diversify


```python
models = []
tree_classifiers = [] #useful to analyze SHAP values later
tuning_num_folds = 2
jobs=4
num_random_state=10
scoring_criteria='accuracy'
predictions = pd.DataFrame()
```


```python
def permutation_importance(fittted_model, XTest, YTest):

    perm_model = PermutationImportance(fittted_model, random_state = num_random_state, cv = 'prefit', scoring="accuracy")
    perm_model.fit(XTest, YTest, scoring="accuracy")

    display(show_weights(perm_model, feature_names = list(XTest.columns)))
```


```python
def check_fitting(model, name):

    plt.figure(figsize = (12,8))

    number_chunks = 5
    train_sizes, train_scores, test_scores = learning_curve(model, X, Y,
                                            train_sizes = np.linspace(0.01, 1.0, number_chunks), cv = 2, scoring = 'accuracy',
                                            n_jobs = -1, random_state = 0)

    train_mean = np.mean(train_scores, axis = 1)
    train_std = np.std(train_scores, axis = 1)

    test_mean = np.mean(test_scores, axis = 1)
    test_std = np.std(test_scores, axis = 1)

    plt.plot(train_sizes, train_mean, '*-', color = 'blue',  label = 'Training score')
    plt.plot(train_sizes, test_mean, '*-', color = 'yellow', label = 'Cross-validation score')

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha = 0.1, color = 'b') # Alpha controls band transparency.
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha = 0.1, color = 'y')

    font_size = 12
    plt.xlabel('Training Set Size', fontsize = font_size)
    plt.ylabel('Accuracy Score', fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.legend(loc = 'best')
    plt.grid()

    plt.suptitle('Check for Overfitting/Underfitting Issues of ' + name + ' When Submited to Different Training Sizes', fontsize = 15)
    plt.tight_layout(rect = [0, 0.03, 1, 0.97])
    Path("./results/census/" + name).mkdir(parents=True, exist_ok=True)

    #saving the image
    plt.savefig('./results/census/' + name + '/learning_curve.png')
    plt.show()
```


```python
# It displays the scores of the combinations of the tunning parameters. It is useful, for example, to see if a parameter is really worth to wasting time to varying it by assess the impact that its variation has in the score
# It can only be applied when we use grid search because we know the combinations made,contrary to randomSearch
def display_tuning_scores(params, tunning):
    keys  = list(params.keys())
    lengths = [len(params[x]) for x in keys]
    if 1 in lengths:
        lengths.remove(1)
    lengths.sort(reverse=True)
    master_scores = tunning.cv_results_['mean_test_score']

    for length in lengths:
        for key, value in params.items():
            if len(value) == length:
                myKey = key
                break

        scores = np.array(master_scores).reshape(tuple(lengths))
        scores = [x.mean() for x in scores]

        plt.figure(figsize=(10,5))
        plt.plot(params[myKey],scores, '*-',)
        plt.xlabel(myKey)
        plt.ylabel('Mean score')
        plt.show()
        params.pop(myKey, None)
        lengths = lengths[1:] + [lengths[0]]
```

## Random Forest Tuning


```python
name = 'Random Forest'
params ={
             'max_depth': st.randint(3, 11),
            'n_estimators': [50,100,250]
            }

""" #due to lack of computational power, the parameters supposed to tune were diminished
params ={
             'max_depth': st.randint(3, 11),
            'n_estimators': [50,100,150,200,250],
             'max_features':["sqrt", "log2"],
             'max_leaf_nodes':st.randint(6, 10)
            }

#remember that the line tunned_model = ... needs to be adjusted depending on the parameters that you include
"""

skf = StratifiedKFold(n_splits=tuning_num_folds, shuffle = True)

random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=params, scoring='roc_auc', n_jobs=jobs, cv=skf.split(X_train,y_train), verbose=3, random_state=num_random_state)
random_search.fit(X_train,y_train)

#These two lines are useful to analyze SHAP values later
tree_classifiers.append(('Random Forest', RandomForestClassifier(n_estimators=random_search.best_params_['n_estimators'], max_depth=random_search.best_params_['max_depth'])))
tree_classifiers.append(('Random Forest', RandomForestClassifier(n_estimators=random_search.best_params_['n_estimators'], max_depth=random_search.best_params_['max_depth'])))

model_config_1 = RandomForestClassifier(n_estimators=random_search.best_params_['n_estimators'], max_depth=random_search.best_params_['max_depth'])
model_config_2 = RandomForestClassifier(n_estimators=random_search.best_params_['n_estimators'], max_depth=random_search.best_params_['max_depth'])
tunned_model = model_config_1.fit(X_train,y_train)
models.append((name, tunned_model))

#let's save the predictions to analyze later their correlation
predictions[name] = tunned_model.predict(X_test)

#Analyze feature importance of Original Data
print(name + " - Feature Importances")
permutation_importance(model_config_2.fit(X_original_train,y_original_train), X_original_test, y_original_test)

#saving model
pickle.dump(tunned_model, open('./models/census/' + name + '_model.sav', 'wb'))
```

    Fitting 2 folds for each of 10 candidates, totalling 20 fits


    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done  20 out of  20 | elapsed:   12.6s remaining:    0.0s
    [Parallel(n_jobs=4)]: Done  20 out of  20 | elapsed:   12.6s finished


    Random Forest - Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1314

                    &plusmn; 0.0017

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 90.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0484

                    &plusmn; 0.0074

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.42%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0328

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.05%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0290

                    &plusmn; 0.0039

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 94.22%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0223

                    &plusmn; 0.0049

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.63%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0150

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.09%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0084

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.27%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0076

                    &plusmn; 0.0023

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.39%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0072

                    &plusmn; 0.0017

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.90%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0053

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                occupation
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.59%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0030

                    &plusmn; 0.0005

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.58%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0005

                    &plusmn; 0.0003

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                race
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.71%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0003

                    &plusmn; 0.0006

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.77%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0002

                    &plusmn; 0.0002

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                native.country
            </td>
        </tr>


    </tbody>
</table>






















## Tuning Logistic Regression


```python
name = 'Logistic Regression'

param = {'penalty': ['l1','l2'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10.0, 100.0, 1000.0]}

""" #due to lack of computational power, the parameters supposed to tune were diminished
C_vals = [0.0001, 0.001, 0.01, 0.1,0.13,0.2, .15, .25, .275, .33, 0.5, .66, 0.75, 1.0, 2.5, 4.0,4.5,5.0,5.1,5.5,6.0, 10.0, 100.0, 1000.0]
penalties = ['l1','l2']
param = {'penalty': penalties, 'C': C_vals}

#remember that the line tunned_model = ... needs to be adjusted depending on the parameters that you include
"""

grid = GridSearchCV(LogisticRegression(), param,verbose=False, cv = StratifiedKFold(n_splits=tuning_num_folds,random_state=num_random_state,shuffle=True), n_jobs=jobs,scoring=scoring_criteria)
grid.fit(X_train,y_train)

model_config_1 = LogisticRegression(penalty=grid.best_params_['penalty'], C=grid.best_params_['C'])
model_config_2 = LogisticRegression(penalty=grid.best_params_['penalty'], C=grid.best_params_['C'])
tunned_model = model_config_1.fit(X_train,y_train)

models.append((name, tunned_model))

#let's save the predictions to analyze later their correlation
predictions[name] = tunned_model.predict(X_test)

#Analyze feature importance of Original Data
print(name + " - Feature Importances")
permutation_importance(model_config_2.fit(X_original_train,y_original_train), X_original_test, y_original_test)

#saving model
pickle.dump(tunned_model, open('./models/census/' + name + '_model.sav', 'wb'))
```

    Logistic Regression - Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0576

                    &plusmn; 0.0074

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 83.05%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0455

                    &plusmn; 0.0019

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 84.93%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0385

                    &plusmn; 0.0038

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 85.16%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0376

                    &plusmn; 0.0020

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 90.65%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0194

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.44%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0172

                    &plusmn; 0.0046

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 94.23%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0098

                    &plusmn; 0.0030

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 94.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0083

                    &plusmn; 0.0018

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.13%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0020

                    &plusmn; 0.0017

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                occupation
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.48%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0003

                    &plusmn; 0.0018

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.51%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0003

                    &plusmn; 0.0009

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                race
            </td>
        </tr>

        <tr style="background-color: hsl(0, 100.00%, 99.70%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0001

                    &plusmn; 0.0015

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(0, 100.00%, 99.60%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0002

                    &plusmn; 0.0003

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                native.country
            </td>
        </tr>

        <tr style="background-color: hsl(0, 100.00%, 99.23%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0006

                    &plusmn; 0.0015

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex
            </td>
        </tr>


    </tbody>
</table>






















## MLPClassifier tuning





```python
name = 'MLP'

param ={
             'activation':['logistic', 'tanh', 'relu'],
             'learning_rate': ['adaptive'],
}

""" #due to lack of computational power, the parameters supposed to tune were diminished
param ={'max_iter': np.logspace(1, 5, 10).astype("int32"),
             'hidden_layer_sizes': np.logspace(2, 3, 4).astype("int32"),
             'activation':['identity', 'logistic', 'tanh', 'relu'],
             'learning_rate': ['adaptive'],
             'early_stopping': [True],
             'alpha': np.logspace(2, 3, 4).astype("int32")
}

#remember that the line tunned_model = ... needs to be adjusted depending on the parameters that you include
"""

grid = GridSearchCV(MLPClassifier(), param,verbose=True, cv = StratifiedKFold(n_splits=tuning_num_folds,random_state=num_random_state,shuffle=True), n_jobs=jobs,scoring=scoring_criteria)
grid.fit(X_train,y_train)

model_config_1 = MLPClassifier(activation=grid.best_params_['activation'], learning_rate=grid.best_params_['learning_rate'])
model_config_2 = MLPClassifier(activation=grid.best_params_['activation'], learning_rate=grid.best_params_['learning_rate'])
tunned_model = model_config_1.fit(X_train,y_train)

models.append((name, tunned_model))

#let's save the predictions to analyze later their correlation
predictions[name] = tunned_model.predict(X_test)

#Analyze feature importance of Original Data
print(name + " - Feature Importances")
permutation_importance(model_config_2.fit(X_original_train,y_original_train), X_original_test, y_original_test)

#display tunning parameters scores
display_tuning_scores(param, grid)

#saving model
pickle.dump(tunned_model, open('./models/census/' + name + '_model.sav', 'wb'))
```

    Fitting 2 folds for each of 3 candidates, totalling 6 fits


    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done   6 out of   6 | elapsed:  1.4min remaining:    0.0s
    [Parallel(n_jobs=4)]: Done   6 out of   6 | elapsed:  1.4min finished


    MLP - Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1030

                    &plusmn; 0.0034

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 84.14%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0739

                    &plusmn; 0.0052

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 88.83%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0448

                    &plusmn; 0.0027

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 90.90%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0334

                    &plusmn; 0.0043

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.81%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0288

                    &plusmn; 0.0015

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.95%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0281

                    &plusmn; 0.0021

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.40%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0258

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.77%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0241

                    &plusmn; 0.0026

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.20%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0096

                    &plusmn; 0.0032

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.89%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0072

                    &plusmn; 0.0020

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                occupation
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.58%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0050

                    &plusmn; 0.0005

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.03%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0038

                    &plusmn; 0.0016

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                race
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.58%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0024

                    &plusmn; 0.0017

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                native.country
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.64%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0022

                    &plusmn; 0.0016

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>


    </tbody>
</table>























![png](outputs/census/output_86_4.png)


## KNeighborsClassifier tuning


```python
name = 'KNN'
params ={'n_neighbors': [5, 15,35],
            'n_neighbors': [1],
            'weights':['uniform','distance']
            }

""" #due to lack of computational power, the parameters supposed to tune were diminished
params ={'n_neighbors': st.randint(5,50),
            'weights':['uniform','distance']
            }
}

#remember that the line tunned_model = ... needs to be adjusted depending on the parameters that you include
"""

skf = StratifiedKFold(n_splits=tuning_num_folds, shuffle = True)

random_search = RandomizedSearchCV(KNeighborsClassifier(), param_distributions=params, scoring='roc_auc', n_jobs=jobs, cv=skf.split(X_train,y_train), verbose=3, random_state=num_random_state)
random_search.fit(X_train,y_train)

model_config_1 = KNeighborsClassifier(n_neighbors=random_search.best_params_['n_neighbors'], weights=random_search.best_params_['weights'])
model_config_2 = KNeighborsClassifier(n_neighbors=random_search.best_params_['n_neighbors'], weights=random_search.best_params_['weights'])
tunned_model = model_config_1.fit(X_train,y_train)

models.append((name, tunned_model))

#let's save the predictions to analyze later their correlation
predictions[name] = tunned_model.predict(X_test)

#Analyze feature importance of Original Data
print(name + " - Feature Importances")
permutation_importance(model_config_2.fit(X_original_train,y_original_train), X_original_test, y_original_test)

#display tunning parameters scores
display_tuning_scores(param, grid)

#saving model
pickle.dump(tunned_model, open('./models/census/' + name + '_model.sav', 'wb'))
```

    Fitting 2 folds for each of 2 candidates, totalling 4 fits


    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done   4 out of   4 | elapsed:  1.5min finished


    KNN - Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1207

                    &plusmn; 0.0039

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 87.62%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0608

                    &plusmn; 0.0046

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 88.43%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0552

                    &plusmn; 0.0034

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 89.39%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0488

                    &plusmn; 0.0050

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 89.79%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0462

                    &plusmn; 0.0049

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 90.41%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0422

                    &plusmn; 0.0049

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 90.86%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0394

                    &plusmn; 0.0045

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                occupation
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.73%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0342

                    &plusmn; 0.0066

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.76%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0340

                    &plusmn; 0.0059

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0322

                    &plusmn; 0.0027

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.75%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0283

                    &plusmn; 0.0017

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.14%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0160

                    &plusmn; 0.0016

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                race
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.57%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0140

                    &plusmn; 0.0027

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.78%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0089

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                native.country
            </td>
        </tr>


    </tbody>
</table>






















## Linear Discriminant tuning


```python
name = 'LDA'

param={'solver': ['svd', 'lsqr', 'eigen']}

grid = GridSearchCV(LinearDiscriminantAnalysis(), param,verbose=True, cv = StratifiedKFold(n_splits=tuning_num_folds,random_state=num_random_state,shuffle=True), n_jobs=jobs,scoring=scoring_criteria)
grid.fit(X_train,y_train)

model_config_1 = LinearDiscriminantAnalysis(solver=grid.best_params_['solver'])
model_config_2 = LinearDiscriminantAnalysis(solver=grid.best_params_['solver'])
tunned_model = model_config_1.fit(X_train,y_train)

models.append((name, tunned_model))

#let's save the predictions to analyze later their correlation
predictions[name] = tunned_model.predict(X_test)

#Analyze feature importance of Original Data
print(name + " - Feature Importances")
permutation_importance(model_config_2.fit(X_original_train,y_original_train), X_original_test, y_original_test)

#display tunning parameters scores
display_tuning_scores(param, grid)

#saving model
pickle.dump(tunned_model, open('./models/census/' + name + '_model.sav', 'wb'))
```

    Fitting 2 folds for each of 3 candidates, totalling 6 fits


    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done   6 out of   6 | elapsed:    8.0s remaining:    0.0s
    [Parallel(n_jobs=4)]: Done   6 out of   6 | elapsed:    8.0s finished


    LDA - Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0624

                    &plusmn; 0.0099

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 84.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0451

                    &plusmn; 0.0050

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 84.36%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0439

                    &plusmn; 0.0023

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.67%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0179

                    &plusmn; 0.0059

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.68%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0148

                    &plusmn; 0.0021

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 94.16%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0107

                    &plusmn; 0.0029

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.34%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0078

                    &plusmn; 0.0021

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.22%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0058

                    &plusmn; 0.0019

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.22%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0006

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                race
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.38%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0004

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.48%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0003

                    &plusmn; 0.0011

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.48%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0003

                    &plusmn; 0.0015

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                occupation
            </td>
        </tr>

        <tr style="background-color: hsl(0, 100.00%, 99.68%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0002

                    &plusmn; 0.0005

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                native.country
            </td>
        </tr>

        <tr style="background-color: hsl(0, 100.00%, 99.27%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0006

                    &plusmn; 0.0013

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education
            </td>
        </tr>


    </tbody>
</table>























![png](outputs/census/output_90_4.png)


## DecisionTreeClassifier tuning

Since each split in the decision tree distinguishes the dependent variable, splits closer to the root, aka starting point, have optimally been determined to have the greatest splitting effect. The feature importance graphic measures how much splitting impact each feature has. It is important to note that this by no means points to causality, but just like in hierarchical clustering, does point to a nebulous groups. Furthermore, for ensemble tree methods, feature impact is aggregated over all the trees.


```python
name = 'DT'
param = {'min_samples_split': [4,7,10,12]}

grid = GridSearchCV(DecisionTreeClassifier(), param,verbose=True, cv = StratifiedKFold(n_splits=tuning_num_folds,random_state=num_random_state,shuffle=True), n_jobs=jobs,scoring=scoring_criteria)
grid.fit(X_train,y_train)

```

    Fitting 2 folds for each of 4 candidates, totalling 8 fits


    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done   8 out of   8 | elapsed:    1.7s finished





    GridSearchCV(cv=StratifiedKFold(n_splits=2, random_state=10, shuffle=True),
                 error_score=nan,
                 estimator=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                                  criterion='gini', max_depth=None,
                                                  max_features=None,
                                                  max_leaf_nodes=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  presort='deprecated',
                                                  random_state=None,
                                                  splitter='best'),
                 iid='deprecated', n_jobs=4,
                 param_grid={'min_samples_split': [4, 7, 10, 12]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=True)




```python
"""
# Helper Function to visualize feature importance if needed
plt.rcParams['figure.figsize'] = (8, 4)
predictors = [x for x in X.columns]
def feature_imp(model):
    MO = model.fit(X_train, y_train)
    feat_imp = pd.Series(MO.feature_importances_, predictors).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
"""

```




    "\n# Helper Function to visualize feature importance if needed\nplt.rcParams['figure.figsize'] = (8, 4)\npredictors = [x for x in X.columns]\ndef feature_imp(model):\n    MO = model.fit(X_train, y_train)\n    feat_imp = pd.Series(MO.feature_importances_, predictors).sort_values(ascending=False)\n    feat_imp.plot(kind='bar', title='Feature Importances')\n    plt.ylabel('Feature Importance Score')\n"



**Pruning the decision trees**

Adapted from [here](https://stackoverflow.com/questions/49428469/pruning-decision-trees)


```python
"""traverse the tree and remove all children of the nodes with minimum class count less than 5  (or any other condition you can think of)."""
def prune_index(inner_tree, index, threshold):
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are children, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)


```


```python
#These two lines are useful to analyze SHAP values later
tree_classifiers.append(('DT', DecisionTreeClassifier(min_samples_split=grid.best_params_['min_samples_split'])))
tree_classifiers.append(('DT', DecisionTreeClassifier(min_samples_split=grid.best_params_['min_samples_split'])))

model_config_1 = DecisionTreeClassifier(min_samples_split=grid.best_params_['min_samples_split'])
model_config_2 = DecisionTreeClassifier(min_samples_split=grid.best_params_['min_samples_split'])
tunned_model = model_config_1.fit(X_train,y_train)

prune_index(tunned_model.tree_, 0, 5)

models.append((name, tunned_model))

#let's save the predictions to analyze later their correlation
predictions[name] = tunned_model.predict(X_test)

#Analyze feature importance of Original Data
print(name + " - Feature Importances")
permutation_importance(model_config_2.fit(X_original_train,y_original_train), X_original_test, y_original_test)

#display tunning parameters scores
display_tuning_scores(param, grid)

#saving model
pickle.dump(tunned_model, open('./models/census/' + name + '_model.sav', 'wb'))
```

    DT - Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1628

                    &plusmn; 0.0049

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 87.20%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0861

                    &plusmn; 0.0090

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 90.05%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0600

                    &plusmn; 0.0043

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 90.11%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0595

                    &plusmn; 0.0056

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.21%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0423

                    &plusmn; 0.0018

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.78%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0307

                    &plusmn; 0.0051

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 94.27%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0273

                    &plusmn; 0.0053

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.36%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0143

                    &plusmn; 0.0050

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                occupation
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.65%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0076

                    &plusmn; 0.0032

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0068

                    &plusmn; 0.0011

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.13%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0055

                    &plusmn; 0.0016

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.25%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0050

                    &plusmn; 0.0041

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.97%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0024

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                race
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.97%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0023

                    &plusmn; 0.0009

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                native.country
            </td>
        </tr>


    </tbody>
</table>























![png](outputs/census/output_97_2.png)


## SVC tuning


```python
name = 'SVC'
param={'C':[0.1,0.5, 1,1.3,1.5],
      'kernel': ["linear","rbf"]
      }

""" #due to lack of computational power, the parameters supposed to tune were diminished
param={'C':[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5],
      'kernel': ["linear","rbf"]
      }

#remember that the line tunned_model = ... needs to be adjusted depending on the parameters that you include
"""

grid = GridSearchCV(SVC(), param,verbose=True, cv = StratifiedKFold(n_splits=tuning_num_folds,random_state=num_random_state,shuffle=True), n_jobs=jobs,scoring=scoring_criteria)
grid.fit(X_train,y_train)

model_config_1 = SVC(C=grid.best_params_['C'], kernel=grid.best_params_['kernel'], probability=True)
model_config_2 = SVC(C=grid.best_params_['C'], kernel=grid.best_params_['kernel'], probability=True)
tunned_model = model_config_1.fit(X_train,y_train)

models.append((name, tunned_model))

#let's save the predictions to analyze later their correlation
predictions[name] = tunned_model.predict(X_test)

#Analyze feature importance of Original Data
print(name + " - Feature Importances")
permutation_importance(model_config_2.fit(X_original_train,y_original_train), X_original_test, y_original_test)

#display tunning parameters scores
display_tuning_scores(param, grid)

#saving model
pickle.dump(tunned_model, open('./models/census/' + name + '_model.sav', 'wb'))
```

    Fitting 2 folds for each of 10 candidates, totalling 20 fits


    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done  20 out of  20 | elapsed:  8.4min finished


    SVC - Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0996

                    &plusmn; 0.0072

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 89.92%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0374

                    &plusmn; 0.0066

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.26%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0305

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.65%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0286

                    &plusmn; 0.0026

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.04%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0267

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.47%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0247

                    &plusmn; 0.0024

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.94%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0181

                    &plusmn; 0.0024

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 94.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0176

                    &plusmn; 0.0035

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.63%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0113

                    &plusmn; 0.0035

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.81%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0072

                    &plusmn; 0.0021

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.43%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0053

                    &plusmn; 0.0018

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                occupation
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.62%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0048

                    &plusmn; 0.0023

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                race
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.36%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0028

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.60%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0022

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                native.country
            </td>
        </tr>


    </tbody>
</table>























![png](outputs/census/output_99_4.png)



![png](outputs/census/output_99_5.png)


## SGDClassifier tuning


```python
name = 'SGD'
param ={'loss':["log","modified_huber"]} #the other available methods do not enable predict_proba that we need later for roc curves and soft voter

""" #due to lack of computational power, the parameters supposed to tune were diminished
param ={'loss':["log","modified_huber","epsilon_insensitive","squared_epsilon_insensitive"]}

#remember that the line tunned_model = ... needs to be adjusted depending on the parameters that you include
"""

grid = GridSearchCV(SGDClassifier(), param,verbose=False, cv = StratifiedKFold(n_splits=tuning_num_folds,random_state=num_random_state,shuffle=True), n_jobs=jobs,scoring=scoring_criteria)
grid.fit(X_train,y_train)

model_config_1 = SGDClassifier(loss=grid.best_params_['loss'])
model_config_2 = SGDClassifier(loss=grid.best_params_['loss'])
tunned_model = model_config_1.fit(X_train,y_train)

models.append((name, tunned_model))

#let's save the predictions to analyze later their correlation
predictions[name] = tunned_model.predict(X_test)

#Analyze feature importance of Original Data
print(name + " - Feature Importances")
permutation_importance(model_config_2.fit(X_original_train,y_original_train), X_original_test, y_original_test)

#display tunning parameters scores
display_tuning_scores(param, grid)

#saving model
pickle.dump(tunned_model, open('./models/census/' + name + '_model.sav', 'wb'))
```

    SGD - Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0579

                    &plusmn; 0.0056

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 81.46%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0520

                    &plusmn; 0.0015

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 85.32%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0372

                    &plusmn; 0.0017

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 85.45%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0368

                    &plusmn; 0.0033

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0129

                    &plusmn; 0.0034

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 94.80%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0085

                    &plusmn; 0.0038

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.38%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0026

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0057

                    &plusmn; 0.0017

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.63%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0013

                    &plusmn; 0.0026

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                occupation
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.11%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0007

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                race
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.30%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0005

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                native.country
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.77%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0001

                    &plusmn; 0.0002

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.86%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0000

                    &plusmn; 0.0007

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education
            </td>
        </tr>

        <tr style="background-color: hsl(0, 100.00%, 97.42%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0031

                    &plusmn; 0.0019

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>


    </tbody>
</table>























![png](outputs/census/output_101_2.png)


## Boosting

### GradientBoostingClassifier tuning


```python
name = 'GraBoost'
param_grid ={
            'loss': ['deviance', 'exponential'],
            'learning_rate': [0.1, 0.01,0.05,0.001],
            }

""" #due to lack of computational power, the parameters supposed to tune were diminished
param_grid ={'n_estimators': st.randint(100, 800),
            'loss': ['deviance', 'exponential'],
            'learning_rate': [0.1, 0.01,0.05,0.001],
            'max_depth': np.arange(2, 12, 1)}
#remember that the line tunned_model = ... needs to be adjusted depending on the parameters that you include
"""

random_search = RandomizedSearchCV(GradientBoostingClassifier(), param_distributions=param_grid, scoring='roc_auc', n_jobs=jobs, cv=skf.split(X_train,y_train), verbose=3, random_state=num_random_state)
random_search.fit(X_train,y_train)

model_config_1 = GradientBoostingClassifier( loss=random_search.best_params_['loss'], learning_rate=random_search.best_params_['learning_rate'], verbose=0)
model_config_2 = GradientBoostingClassifier( loss=random_search.best_params_['loss'], learning_rate=random_search.best_params_['learning_rate'], verbose=0)
tunned_model = model_config_1.fit(X_train,y_train)

models.append((name, tunned_model))

#let's save the predictions to analyze later their correlation
predictions[name] = tunned_model.predict(X_test)

#Analyze feature importance of Original Data
print(name + " - Feature Importances")
permutation_importance(model_config_2.fit(X_original_train,y_original_train), X_original_test, y_original_test)


#saving model
pickle.dump(tunned_model, open('./models/census/' + name + '_model.sav', 'wb'))
```

    Fitting 2 folds for each of 8 candidates, totalling 16 fits


    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done  16 out of  16 | elapsed:  1.5min finished


    GraBoost - Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0633

                    &plusmn; 0.0048

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 81.17%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0581

                    &plusmn; 0.0071

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 83.44%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0483

                    &plusmn; 0.0025

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 84.23%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0450

                    &plusmn; 0.0023

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 84.48%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0440

                    &plusmn; 0.0031

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 90.32%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0224

                    &plusmn; 0.0041

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 90.70%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0212

                    &plusmn; 0.0029

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.08%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0085

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.65%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0072

                    &plusmn; 0.0017

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.63%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0050

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.89%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0044

                    &plusmn; 0.0018

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                occupation
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.17%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0007

                    &plusmn; 0.0006

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                native.country
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.34%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0005

                    &plusmn; 0.0006

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                race
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.65%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0002

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>


    </tbody>
</table>






















### AdaBoostClassifier tuning


```python
name = 'AdaBoost'
params ={'n_estimators':st.randint(100, 400),
        'learning_rate':np.arange(.1, 4, .5)}

""" #due to lack of computational power, the parameters supposed to tune were diminished
params ={'n_estimators':st.randint(100, 800),
        'learning_rate':np.arange(.1, 4, .5)}

#remember that the line tunned_model = ... needs to be adjusted depending on the parameters that you include
"""

random_search = RandomizedSearchCV(AdaBoostClassifier(), param_distributions=params, scoring='roc_auc', n_jobs=jobs, cv=skf.split(X_train,y_train), verbose=3, random_state=num_random_state)
random_search.fit(X_train,y_train)

model_config_1 = AdaBoostClassifier(n_estimators=random_search.best_params_['n_estimators'], learning_rate=random_search.best_params_['learning_rate'])
model_config_2 = AdaBoostClassifier(n_estimators=random_search.best_params_['n_estimators'], learning_rate=random_search.best_params_['learning_rate'])
tunned_model = model_config_1.fit(X_train,y_train)

models.append((name, tunned_model))

#let's save the predictions to analyze later their correlation
predictions[name] = tunned_model.predict(X_test)

#Analyze feature importance of Original Data
print(name + " - Feature Importances")
permutation_importance(model_config_2.fit(X_original_train,y_original_train), X_original_test, y_original_test)

#saving model
pickle.dump(tunned_model, open('./models/census/' + name + '_model.sav', 'wb'))
```

    Fitting 2 folds for each of 10 candidates, totalling 20 fits


    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done  20 out of  20 | elapsed:  3.3min remaining:    0.0s
    [Parallel(n_jobs=4)]: Done  20 out of  20 | elapsed:  3.3min finished


    AdaBoost - Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0788

                    &plusmn; 0.0031

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 81.47%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0706

                    &plusmn; 0.0042

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 85.86%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0480

                    &plusmn; 0.0018

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 86.13%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0467

                    &plusmn; 0.0037

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 89.55%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0312

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.23%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0168

                    &plusmn; 0.0024

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 94.70%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0118

                    &plusmn; 0.0024

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.57%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0092

                    &plusmn; 0.0015

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.12%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0076

                    &plusmn; 0.0029

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                occupation
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.49%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0041

                    &plusmn; 0.0019

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.32%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0023

                    &plusmn; 0.0016

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                native.country
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.34%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0006

                    &plusmn; 0.0018

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.42%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0005

                    &plusmn; 0.0018

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                race
            </td>
        </tr>

        <tr style="background-color: hsl(0, 100.00%, 99.52%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0004

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship
            </td>
        </tr>


    </tbody>
</table>






















### XGBClassifier Tuning


```python
name = 'XGB'
params = {
        'gamma': [0.5, 1, 1.5, 2, 5],
        'max_depth': [3, 4, 5]
        }

""" #due to lack of computational power, the parameters supposed to tune were diminished
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
#remember that the line tunned_model = ... needs to be adjusted depending on the parameters that you include
"""


skf = StratifiedKFold(n_splits=tuning_num_folds, shuffle = True)

random_search = RandomizedSearchCV(XGBClassifier(), param_distributions=params, scoring='roc_auc', n_jobs=jobs, cv=skf.split(X_train,y_train), verbose=3, random_state=num_random_state)
random_search.fit(X_train,y_train)

model_config_1 = XGBClassifier(gamma=random_search.best_params_['gamma'], max_depth=random_search.best_params_['max_depth'], verbose=0)
model_config_2 = XGBClassifier(gamma=random_search.best_params_['gamma'], max_depth=random_search.best_params_['max_depth'], verbose=0)
tunned_model = model_config_1.fit(X_train,y_train)

models.append((name, tunned_model))

#let's save the predictions to analyze later their correlation
predictions[name] = tunned_model.predict(X_test)

#Analyze feature importance of Original Data
print(name + " - Feature Importances")
permutation_importance(model_config_2.fit(X_original_train,y_original_train), X_original_test, y_original_test)

#saving model
pickle.dump(tunned_model, open('./models/census/' + name + '_model.sav', 'wb'))
```

    Fitting 2 folds for each of 10 candidates, totalling 20 fits


    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done  20 out of  20 | elapsed:  3.0min remaining:    0.0s
    [Parallel(n_jobs=4)]: Done  20 out of  20 | elapsed:  3.0min finished


    XGB - Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0852

                    &plusmn; 0.0045

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 80.04%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0850

                    &plusmn; 0.0040

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 85.19%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0555

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 86.41%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0491

                    &plusmn; 0.0021

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 87.45%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0438

                    &plusmn; 0.0005

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.52%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0250

                    &plusmn; 0.0024

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.45%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0212

                    &plusmn; 0.0030

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.78%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0092

                    &plusmn; 0.0023

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.51%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0070

                    &plusmn; 0.0020

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.54%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0070

                    &plusmn; 0.0014

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.81%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0062

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                occupation
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.15%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0009

                    &plusmn; 0.0011

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                native.country
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.45%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0005

                    &plusmn; 0.0014

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.63%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0003

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                race
            </td>
        </tr>


    </tbody>
</table>























```python
name = 'Naive Bayes'
model_config_1 = GaussianNB()
model_config_2 = GaussianNB()
tunned_model = model_config_1.fit(X_train,y_train)

models.append((name, tunned_model))

#let's save the predictions to analyze later their correlation
predictions[name] = tunned_model.predict(X_test)

#Analyze feature importance of Original Data
print(name + " - Feature Importances")
permutation_importance(model_config_2.fit(X_original_train,y_original_train), X_original_test, y_original_test)

#saving model
pickle.dump(tunned_model, open('./models/census/' + name + '_model.sav', 'wb'))
```

    Naive Bayes - Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0705

                    &plusmn; 0.0061

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 88.30%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0328

                    &plusmn; 0.0019

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 89.82%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0269

                    &plusmn; 0.0038

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 90.29%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0251

                    &plusmn; 0.0030

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 90.57%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0241

                    &plusmn; 0.0050

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.46%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0209

                    &plusmn; 0.0029

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.59%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0171

                    &plusmn; 0.0027

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.48%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0142

                    &plusmn; 0.0024

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.05%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0096

                    &plusmn; 0.0031

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.96%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0027

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                occupation
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.20%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0023

                    &plusmn; 0.0003

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                race
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.42%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0019

                    &plusmn; 0.0024

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass
            </td>
        </tr>

        <tr style="background-color: hsl(0, 100.00%, 99.93%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0000

                    &plusmn; 0.0014

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(0, 100.00%, 98.68%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0014

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                native.country
            </td>
        </tr>


    </tbody>
</table>






















### CatBoostClassifier Tuning


```python
name = 'CAT'
param = {'iterations': [100, 150], 'learning_rate': [0.3, 0.4, 0.5]}

grid = GridSearchCV(CatBoostClassifier(), param,verbose=True, cv = StratifiedKFold(n_splits=tuning_num_folds,random_state=num_random_state,shuffle=True), n_jobs=jobs,scoring=scoring_criteria)
grid.fit(X_train,y_train)

model_config_1 = CatBoostClassifier(iterations=grid.best_params_['iterations'], learning_rate=grid.best_params_['learning_rate'], verbose=0)
model_config_2 = CatBoostClassifier(iterations=grid.best_params_['iterations'], learning_rate=grid.best_params_['learning_rate'], verbose=0)
tunned_model = model_config_1.fit(X_train,y_train, verbose=0)

models.append((name, tunned_model))

#let's save the predictions to analyze later their correlation
predictions[name] = tunned_model.predict(X_test)

#Analyze feature importance of Original Data
print(name + " - Feature Importances")
permutation_importance(model_config_2.fit(X_original_train,y_original_train), X_original_test, y_original_test)

#display tunning parameters scores
display_tuning_scores(param, grid)

#saving model
pickle.dump(tunned_model, open('./models/census/' + name + '_model.sav', 'wb'))
```

    Fitting 2 folds for each of 6 candidates, totalling 12 fits


    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done  12 out of  12 | elapsed:  1.5min finished


    0:	learn: 0.4729683	total: 88.1ms	remaining: 13.1s
    1:	learn: 0.3915143	total: 130ms	remaining: 9.6s
    2:	learn: 0.3534348	total: 167ms	remaining: 8.17s
    3:	learn: 0.3301681	total: 188ms	remaining: 6.87s
    4:	learn: 0.3187881	total: 285ms	remaining: 8.26s
    5:	learn: 0.3112287	total: 323ms	remaining: 7.75s
    6:	learn: 0.2996086	total: 349ms	remaining: 7.13s
    7:	learn: 0.2849331	total: 396ms	remaining: 7.02s
    8:	learn: 0.2802473	total: 428ms	remaining: 6.7s
    9:	learn: 0.2767624	total: 449ms	remaining: 6.29s
    10:	learn: 0.2664189	total: 475ms	remaining: 6s
    11:	learn: 0.2645821	total: 505ms	remaining: 5.81s
    12:	learn: 0.2605946	total: 525ms	remaining: 5.53s
    13:	learn: 0.2578862	total: 551ms	remaining: 5.36s
    14:	learn: 0.2561950	total: 606ms	remaining: 5.46s
    15:	learn: 0.2540116	total: 724ms	remaining: 6.07s
    16:	learn: 0.2519333	total: 825ms	remaining: 6.46s
    17:	learn: 0.2494914	total: 877ms	remaining: 6.43s
    18:	learn: 0.2480393	total: 934ms	remaining: 6.44s
    19:	learn: 0.2468824	total: 972ms	remaining: 6.32s
    20:	learn: 0.2432083	total: 1000ms	remaining: 6.14s
    21:	learn: 0.2421580	total: 1.02s	remaining: 5.94s
    22:	learn: 0.2409310	total: 1.05s	remaining: 5.82s
    23:	learn: 0.2399049	total: 1.12s	remaining: 5.88s
    24:	learn: 0.2389909	total: 1.15s	remaining: 5.77s
    25:	learn: 0.2381301	total: 1.18s	remaining: 5.63s
    26:	learn: 0.2371352	total: 1.21s	remaining: 5.49s
    27:	learn: 0.2366029	total: 1.25s	remaining: 5.46s
    28:	learn: 0.2353401	total: 1.29s	remaining: 5.37s
    29:	learn: 0.2337220	total: 1.32s	remaining: 5.3s
    30:	learn: 0.2330665	total: 1.36s	remaining: 5.23s
    31:	learn: 0.2323586	total: 1.39s	remaining: 5.14s
    32:	learn: 0.2316171	total: 1.44s	remaining: 5.12s
    33:	learn: 0.2301754	total: 1.48s	remaining: 5.06s
    34:	learn: 0.2292186	total: 1.51s	remaining: 4.97s
    35:	learn: 0.2282726	total: 1.54s	remaining: 4.88s
    36:	learn: 0.2278894	total: 1.6s	remaining: 4.88s
    37:	learn: 0.2269561	total: 1.64s	remaining: 4.83s
    38:	learn: 0.2254705	total: 1.67s	remaining: 4.75s
    39:	learn: 0.2247264	total: 1.7s	remaining: 4.66s
    40:	learn: 0.2241728	total: 1.76s	remaining: 4.69s
    41:	learn: 0.2237808	total: 1.84s	remaining: 4.72s
    42:	learn: 0.2235096	total: 1.89s	remaining: 4.7s
    43:	learn: 0.2229428	total: 1.92s	remaining: 4.62s
    44:	learn: 0.2222835	total: 1.98s	remaining: 4.61s
    45:	learn: 0.2216415	total: 2.02s	remaining: 4.57s
    46:	learn: 0.2210691	total: 2.07s	remaining: 4.53s
    47:	learn: 0.2198521	total: 2.1s	remaining: 4.46s
    48:	learn: 0.2178722	total: 2.12s	remaining: 4.37s
    49:	learn: 0.2170893	total: 2.18s	remaining: 4.36s
    50:	learn: 0.2150661	total: 2.21s	remaining: 4.29s
    51:	learn: 0.2143972	total: 2.24s	remaining: 4.21s
    52:	learn: 0.2139624	total: 2.29s	remaining: 4.18s
    53:	learn: 0.2135379	total: 2.31s	remaining: 4.1s
    54:	learn: 0.2131996	total: 2.33s	remaining: 4.03s
    55:	learn: 0.2127256	total: 2.37s	remaining: 3.98s
    56:	learn: 0.2121261	total: 2.41s	remaining: 3.93s
    57:	learn: 0.2117372	total: 2.43s	remaining: 3.86s
    58:	learn: 0.2114055	total: 2.47s	remaining: 3.81s
    59:	learn: 0.2108824	total: 2.5s	remaining: 3.75s
    60:	learn: 0.2099241	total: 2.52s	remaining: 3.68s
    61:	learn: 0.2094251	total: 2.57s	remaining: 3.65s
    62:	learn: 0.2089940	total: 2.6s	remaining: 3.6s
    63:	learn: 0.2084980	total: 2.63s	remaining: 3.53s
    64:	learn: 0.2081596	total: 2.68s	remaining: 3.51s
    65:	learn: 0.2069614	total: 2.71s	remaining: 3.45s
    66:	learn: 0.2065474	total: 2.73s	remaining: 3.39s
    67:	learn: 0.2059148	total: 2.78s	remaining: 3.35s
    68:	learn: 0.2054869	total: 2.81s	remaining: 3.29s
    69:	learn: 0.2050633	total: 2.84s	remaining: 3.24s
    70:	learn: 0.2046455	total: 2.86s	remaining: 3.19s
    71:	learn: 0.2044274	total: 2.9s	remaining: 3.14s
    72:	learn: 0.2040614	total: 2.94s	remaining: 3.1s
    73:	learn: 0.2037156	total: 2.98s	remaining: 3.06s
    74:	learn: 0.2034493	total: 3.01s	remaining: 3.01s
    75:	learn: 0.2028351	total: 3.08s	remaining: 3s
    76:	learn: 0.2024786	total: 3.19s	remaining: 3.02s
    77:	learn: 0.2022114	total: 3.24s	remaining: 3s
    78:	learn: 0.2018264	total: 3.34s	remaining: 3s
    79:	learn: 0.2014151	total: 3.44s	remaining: 3.01s
    80:	learn: 0.2012671	total: 3.48s	remaining: 2.97s
    81:	learn: 0.2008989	total: 3.51s	remaining: 2.91s
    82:	learn: 0.2003869	total: 3.58s	remaining: 2.89s
    83:	learn: 0.2000606	total: 3.63s	remaining: 2.85s
    84:	learn: 0.1998027	total: 3.66s	remaining: 2.8s
    85:	learn: 0.1991828	total: 3.68s	remaining: 2.74s
    86:	learn: 0.1986930	total: 3.71s	remaining: 2.68s
    87:	learn: 0.1980936	total: 3.75s	remaining: 2.64s
    88:	learn: 0.1977639	total: 3.77s	remaining: 2.59s
    89:	learn: 0.1975471	total: 3.85s	remaining: 2.56s
    90:	learn: 0.1970148	total: 3.87s	remaining: 2.51s
    91:	learn: 0.1966961	total: 3.91s	remaining: 2.46s
    92:	learn: 0.1961422	total: 3.93s	remaining: 2.41s
    93:	learn: 0.1959543	total: 3.98s	remaining: 2.37s
    94:	learn: 0.1957163	total: 4.01s	remaining: 2.32s
    95:	learn: 0.1956455	total: 4.08s	remaining: 2.29s
    96:	learn: 0.1953173	total: 4.19s	remaining: 2.29s
    97:	learn: 0.1949937	total: 4.22s	remaining: 2.24s
    98:	learn: 0.1947543	total: 4.28s	remaining: 2.2s
    99:	learn: 0.1944004	total: 4.31s	remaining: 2.16s
    100:	learn: 0.1941151	total: 4.34s	remaining: 2.11s
    101:	learn: 0.1939073	total: 4.39s	remaining: 2.07s
    102:	learn: 0.1934292	total: 4.43s	remaining: 2.02s
    103:	learn: 0.1925899	total: 4.47s	remaining: 1.98s
    104:	learn: 0.1923747	total: 4.5s	remaining: 1.93s
    105:	learn: 0.1919536	total: 4.57s	remaining: 1.9s
    106:	learn: 0.1917642	total: 4.6s	remaining: 1.85s
    107:	learn: 0.1914438	total: 4.64s	remaining: 1.8s
    108:	learn: 0.1908900	total: 4.7s	remaining: 1.77s
    109:	learn: 0.1905485	total: 4.73s	remaining: 1.72s
    110:	learn: 0.1901081	total: 4.75s	remaining: 1.67s
    111:	learn: 0.1896557	total: 4.8s	remaining: 1.63s
    112:	learn: 0.1893759	total: 4.84s	remaining: 1.59s
    113:	learn: 0.1890558	total: 4.87s	remaining: 1.54s
    114:	learn: 0.1886892	total: 4.91s	remaining: 1.49s
    115:	learn: 0.1880756	total: 4.93s	remaining: 1.44s
    116:	learn: 0.1875747	total: 4.95s	remaining: 1.4s
    117:	learn: 0.1871418	total: 4.99s	remaining: 1.35s
    118:	learn: 0.1869238	total: 5.04s	remaining: 1.31s
    119:	learn: 0.1866486	total: 5.13s	remaining: 1.28s
    120:	learn: 0.1863621	total: 5.18s	remaining: 1.24s
    121:	learn: 0.1861017	total: 5.23s	remaining: 1.2s
    122:	learn: 0.1858232	total: 5.25s	remaining: 1.15s
    123:	learn: 0.1854372	total: 5.29s	remaining: 1.11s
    124:	learn: 0.1850747	total: 5.31s	remaining: 1.06s
    125:	learn: 0.1847413	total: 5.34s	remaining: 1.02s
    126:	learn: 0.1844416	total: 5.36s	remaining: 971ms
    127:	learn: 0.1840402	total: 5.39s	remaining: 927ms
    128:	learn: 0.1837779	total: 5.41s	remaining: 882ms
    129:	learn: 0.1835798	total: 5.44s	remaining: 837ms
    130:	learn: 0.1833935	total: 5.46s	remaining: 793ms
    131:	learn: 0.1830063	total: 5.49s	remaining: 748ms
    132:	learn: 0.1827289	total: 5.54s	remaining: 708ms
    133:	learn: 0.1825402	total: 5.57s	remaining: 666ms
    134:	learn: 0.1823325	total: 5.6s	remaining: 623ms
    135:	learn: 0.1821067	total: 5.63s	remaining: 579ms
    136:	learn: 0.1819330	total: 5.68s	remaining: 539ms
    137:	learn: 0.1816996	total: 5.72s	remaining: 498ms
    138:	learn: 0.1814379	total: 5.75s	remaining: 455ms
    139:	learn: 0.1812804	total: 5.78s	remaining: 413ms
    140:	learn: 0.1810492	total: 5.83s	remaining: 372ms
    141:	learn: 0.1808818	total: 5.87s	remaining: 331ms
    142:	learn: 0.1807497	total: 5.9s	remaining: 289ms
    143:	learn: 0.1804044	total: 5.95s	remaining: 248ms
    144:	learn: 0.1801411	total: 5.99s	remaining: 206ms
    145:	learn: 0.1797765	total: 6.02s	remaining: 165ms
    146:	learn: 0.1794787	total: 6.04s	remaining: 123ms
    147:	learn: 0.1791834	total: 6.09s	remaining: 82.3ms
    148:	learn: 0.1790279	total: 6.11s	remaining: 41ms
    149:	learn: 0.1788011	total: 6.13s	remaining: 0us
    CAT - Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0887

                    &plusmn; 0.0042

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 80.77%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0839

                    &plusmn; 0.0032

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 81.50%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0794

                    &plusmn; 0.0042

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 83.26%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0688

                    &plusmn; 0.0042

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 87.62%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0447

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.64%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0213

                    &plusmn; 0.0040

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.85%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0165

                    &plusmn; 0.0029

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.93%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0161

                    &plusmn; 0.0031

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.86%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0093

                    &plusmn; 0.0021

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.02%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0088

                    &plusmn; 0.0027

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                occupation
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.30%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0080

                    &plusmn; 0.0017

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.13%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0010

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                native.country
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.20%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0009

                    &plusmn; 0.0018

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 99.56%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0004

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                race
            </td>
        </tr>


    </tbody>
</table>























![png](outputs/census/output_111_4.png)



![png](outputs/census/output_111_5.png)


## Ensemble Methods

**Studying Correlation**
If base models' predictions are weakly correlated with each other, the ensemble will likely to perform better. On the other hand, for a strong correlation of predictions among the base models, the ensemble will unlikely to perform better.


```python
plt.figure(figsize = (20, 6))
correlations = predictions.corr()
sns.heatmap(correlations, annot = True)
plt.title('Prediction Correlation in Simpler Classifiers', fontsize = 18)
plt.show()
```


![png](outputs/census/output_114_0.png)


**As we can see, there are some classifiers that have a high correlated predictions. So, we won't use them to form the ensemble models.**


```python
classifiers = copy.deepcopy(models)
every_model = copy.deepcopy(models)

for i in range(0, len(correlations.columns)):
    for j in range(0, len(correlations.columns)):
        if j>=i:
            break
        else:
            if abs(correlations.iloc[i,j]) >= 0.85:
                for index, clf in enumerate(classifiers):
                    if correlations.columns[j] == clf[0]:
                        del classifiers[index]

classifiers = [i for i in classifiers]
```

### Voting

Voting can be subdivided into hard voting and soft voting.

Hard voting is purely atributing the majority class from the predictions of all the classifiers.

Soft voting is useful when we have probabilities as output of the classifiers, so we average those values and assume the class that it's close to the probability.

We are going to do just soft voting because it takes into account how certain each voter is, rather than just a binary input from the voter...so, theoretically, it's better.


```python
def soft_voting(average_results, models_test_accuracies, models_train_accuracies, models_validation_accuracies, X_train, X_test, Y_train, Y_test, seed, num_folds, scoring):
    copy_1 = average_results[average_results.Model.isin([i[0] for i in classifiers])]
    copy_1.sort_values(by=["Test Accuracy"], ascending=False, inplace=True)

    numb_voters = np.arange(2,len(classifiers)+1,1)

    #we we'll try to ensemble the best two classifiers, then the best three, the best, four, etc
    for i in numb_voters:
        models_names = list(copy_1.iloc[0:i,0])
        voting_models = []
        for model1 in models_names:
            for model2 in models:
                if model2[0] == model1:
                    voting_models.append(model2[1])

        model_config_1 = EnsembleVoteClassifier(clfs = voting_models, voting = 'soft')
        model_config_2 = EnsembleVoteClassifier(clfs = voting_models, voting = 'soft')
        model = model_config_1.fit(X_train,Y_train)

        name = "SoftVoter_" + str(i) + "_best"

        every_model.append((name, model))
        pickle.dump(model, open('./models/census/' + name + '_model.sav', 'wb'))
        average_results,models_test_accuracies, models_train_accuracies, models_validation_accuracies = classify_performance(name, model, X_train, X_test, Y_train, Y_test, num_folds, scoring, seed, average_results, models_test_accuracies, models_train_accuracies, models_validation_accuracies)

    return average_results,models_test_accuracies, models_train_accuracies, models_validation_accuracies
```
#To apply Hard Voting, here's the code anyway
"""
def hard_voting(average_results, models_test_accuracies, models_train_accuracies, models_validation_accuracies, X_train, X_test, Y_train, Y_test, seed, num_folds, scoring):

    copy_1 = average_results[average_results.Model.isin([i[0] for i in classifiers])]
    copy_1.sort_values(by=["Test Accuracy"], ascending=False, inplace=True)

    classifiers_voting = classifiers[:-1]

    #Hard-Voting
    numb_voters = np.arange(2,len(classifiers_voting)+1,1) #if catboost counted, it would be len(classifiers)+1

    all_test_accuracy = []
    all_y_pred = []

    #we we'll try to ensemble the best two classifiers, then the best three, the best, four, etc
    for i in numb_voters:
        models_names = list(copy_1.iloc[0:i,0])
        voting_models = []
        for model1 in models_names:
            for model2 in models[:-1]:
                if model2[0] == model1:
                    voting_models.append(model2[1])

        model_config_1 = EnsembleVoteClassifier(clfs = voting_models, voting = 'hard')
        model_config_2 = EnsembleVoteClassifier(clfs = voting_models, voting = 'hard')
        model = model_config_1.fit(X_train,Y_train)

        name = "HardVoter_" + str(i) + "_best"

        every_model.append((name, model))
        pickle.dump(model, open('./models/census/' + name + '_model.sav', 'wb'))
        average_results,models_test_accuracies, models_train_accuracies, models_validation_accuracies = classify_performance(name, model, X_train, X_test, Y_train, Y_test, num_folds, scoring, seed, average_results, models_test_accuracies, models_train_accuracies, models_validation_accuracies)

    return average_results,models_test_accuracies, models_train_accuracies, models_validation_accuracies
"""
### Bagging (Bootstrapping Aggregation)

Bagging is characteristic of random forest. You bootstrap or subdivide the same training set into multiple subsets or bags. This steps is also called row sampling with replacement. Each of these training bags will feed a mmodel to be trained. After each model being trained, it is time for aggragation, in other orders, to predict the outcome based on a voting system of each model trained.


```python
def bagging(average_results, models_test_accuracies, models_train_accuracies, models_validation_accuracies, X_train, X_test, Y_train, Y_test, seed, num_folds, scoring):

    random_forest_classifier = RandomForestClassifier() # is the random forest classifier

    model_config_1 = BaggingClassifier(base_estimator = random_forest_classifier, verbose = 0, n_jobs = -1, random_state = seed)
    model_config_2 = BaggingClassifier(base_estimator = random_forest_classifier, verbose = 0, n_jobs = -1, random_state = seed)

    model = model_config_1.fit(X_train, Y_train)
    name = "bagging"

    every_model.append((name, model))
    pickle.dump(model, open('./models/census/' + name + '_model.sav', 'wb'))

    return classify_performance(name, model, X_train, X_test, Y_train, Y_test, num_folds, scoring, seed, average_results, models_test_accuracies, models_train_accuracies, models_validation_accuracies)


```

### Blending


```python
def blending(average_results, models_test_accuracies, models_train_accuracies, models_validation_accuracies, X_train, X_test, Y_train, Y_test, seed, num_folds, scoring):

    copy_2 = average_results[average_results.Model.isin([i[0] for i in classifiers])]
    copy_2.sort_values(by=["Test Accuracy"], ascending=False, inplace=True)

    numb_voters = np.arange(2,len(classifiers)+1,1)

    #we we'll try to ensemble the best two classifiers, then the best three, the best, four, etc
    for i in numb_voters:
        models_names = list(copy_2.iloc[0:i,0])
        blending_models = []
        for model1 in models_names:
            for model2 in models:
                if model2[0] == model1:
                    blending_models.append(model2[1])

        blend_1 = BlendEnsemble(n_jobs = -1, test_size = 0.5, random_state = seed, scorer="accuracy")
        blend_1.add(blending_models, proba=True)
        blend_1.add_meta(RandomForestClassifier())

        blend_2 = BlendEnsemble(n_jobs = -1, test_size = 0.5, random_state = seed)
        blend_2.add(blending_models, proba=True) #proba=true s important in order to predict_proba function properly later
        blend_2.add_meta(RandomForestClassifier())

        model = blend_1.fit(X_train, Y_train, scoring="accuracy")
        name = "blending_" + str(i) + "_best"

        every_model.append((name, model))
        pickle.dump(model, open('./models/census/' + name + '_model.sav', 'wb'))

        average_results,models_test_accuracies, models_train_accuracies, models_validation_accuracies = classify_performance(name, model, X_train, X_test, Y_train, Y_test, num_folds, scoring, seed, average_results, models_test_accuracies, models_train_accuracies, models_validation_accuracies)


    return average_results,models_test_accuracies, models_train_accuracies, models_validation_accuracies

```

### Stacking

Shortly, we are trying to fit a model upon Y_valid data and models' predictions (stacked) made during the validation stage. Then, we will try to predict from the models' predictions (stacked) of test data to see if we reach Y_test values.

In this model we can't analyze the true feature importance because all the features that matter are the predictions of the classifiers!


```python
def stacking_function(average_results, models_test_accuracies, models_train_accuracies, models_validation_accuracies, X_train, X_test, Y_train, Y_test, seed, num_folds, scoring):
    numb_voters = np.arange(2,len(classifiers)+1, 1)

    final_classifiers = copy.deepcopy(classifiers)
    final_classifiers = [i[1] for i in final_classifiers]

    myStack = StackingTransformer(final_classifiers, X_train, Y_train, X_test, regression = False, mode = 'oof_pred_bag', needs_proba = False, save_dir = None,metric = accuracy_score, n_folds = num_folds, stratified = True, shuffle = True, random_state =  seed, verbose = 0)
    myStack.fit(X_train, Y_train)
    S_train = myStack.transform(X_train)
    S_test = myStack.transform(X_test)
    joblib.dump(myStack, open('./models/census/stack_transformer.pkl', 'wb'))

    super_learner = RandomForestClassifier()
    model_1 = super_learner.fit(S_train, Y_train)
    name = "stacking"

    every_model.append((name, model))
    pickle.dump(model, open('./models/census/' + name + '_model.sav', 'wb'))

    return classify_performance(name, model_1, S_train, S_test, Y_train, Y_test, num_folds, scoring, seed, average_results, models_test_accuracies, models_train_accuracies, models_validation_accuracies)

```

# Runs
**First of all, we'll execute 30 tests of cross-validation of all the models but with different seeds.**


```python
def classify_performance(name, model, X_train, X_test, Y_train, Y_test, num_folds, scoring, seed, average_results, models_test_accuracies, models_train_accuracies, models_validation_accuracies):
    stratifiedfold = StratifiedShuffleSplit(n_splits=num_folds, random_state=seed)

    cv_results = cross_validate(model, X_train, Y_train, cv=stratifiedfold, scoring=scoring, verbose=0, return_train_score=True)

    print("%s - Train: Accuracy mean - %.3f, Precision: %.3f, Recall: %.3f." % (name, cv_results['train_accuracy'].mean(), cv_results['train_precision'].mean(), cv_results['train_recall'].mean()))

    pred_proba = model.predict_proba(X_test)
    if (pred_proba[0].size == 1 ): #just in case that probabilities are not correctly retrieved
        pred_proba = pred_proba.T
    else:
        pred_proba = pred_proba[:, 1]

    y_pred =  [round(x) for x in pred_proba]

    precision, recall, fscore, support = precision_recall_fscore_support(Y_test, y_pred,average='binary')
    test_accuracy = accuracy_score(Y_test, y_pred)

    print("%s - Test: Accuracy - %.3f, Precision: %.3f, Recall: %.3f, F-Score: %.3f" % (name, test_accuracy, precision,recall, fscore))

    if (seed == 0):
        print(name + " - Learning Curve")
        check_fitting(model, name)

    fpr, tpr, thresholds1 = roc_curve(Y_test, pred_proba)
    pre, rec, thresholds2 = precision_recall_curve(Y_test, pred_proba)
    roc_auc = auc(fpr, tpr)

    TP = FP = TN = FN = 0
    for i in range(len(y_pred)):
        if ((y_pred[i] == 1) and (list(Y_test)[i] == y_pred[i])):
            TP += 1
        if y_pred[i]==1 and list(Y_test)[i]!=y_pred[i]:
            FP += 1
        if y_pred[i]==0 and list(Y_test)[i]==y_pred[i]:
            TN += 1
        if y_pred[i]==0 and list(Y_test)[i]!=y_pred[i]:
            FN += 1

    new_entry = {'Model': name,
                 'Test Accuracy': test_accuracy,
                 'Test Precision': precision,
                 'Test Recall': recall,
                 'Test F-Score': fscore,
                 'AUC': roc_auc,
                 'TPR': tpr,
                 'FPR': fpr,
                 'TP': TP,
                 'FP': FP,
                 'TN': TN,
                 'FN': FN,
                 'Train Accuracy':cv_results['train_accuracy'].mean(),
                 'Train Precision': cv_results['train_precision'].mean(),
                 'Train Recall': cv_results['train_recall'].mean(),
                 'Validation Accuracy':cv_results['test_accuracy'].mean(),
                 'Validation Precision': cv_results['test_precision'].mean(),
                 'Validation Recall': cv_results['test_recall'].mean(),
                 'PRE': pre,
                 'REC': rec
                }


    for key, value in new_entry.items():
        if key != 'Model':

            if key == 'TPR' or key == 'FPR' or key == 'PRE' or key =='REC':
                current = average_results.ix[average_results['Model'] == name][key].values

                if type(current[0]) == int:
                    average_results.iloc[average_results.ix[average_results['Model'] == name].index.values.astype(int)[0]][key] = value
                else:
                    summed = list(map(add, current[0], list(value)))
                    average_results.iloc[average_results.ix[average_results['Model'] == name].index.values.astype(int)[0]][key] = summed
            else:

                average_results.iloc[average_results.ix[average_results['Model'] == name].index.values.astype(int)[0]][key] = float(average_results.ix[average_results['Model'] == name][key].values) + value

            models_test_accuracies.iloc[seed, :][name] = test_accuracy
            models_train_accuracies.iloc[seed, :][name] = cv_results['train_accuracy'].mean()
            models_validation_accuracies.iloc[seed, :][name] = cv_results['test_accuracy'].mean()

    return average_results, models_test_accuracies, models_train_accuracies, models_validation_accuracies

```


```python
# RUNNING ALL THE EXPERIMENTS
###
num_experiments = 4
num_folds = 2
scoring = {'accuracy': 'accuracy',
           'precision': 'precision_macro',
           'recall': 'recall_macro'}


all_models = [i[0] for i in models]

for i in np.arange(2,len(classifiers)+1,1):
              all_models.append("blending_" + str(i) + "_best")

all_models.append("bagging")

for i in np.arange(2,len(classifiers)+1,1):
              all_models.append("SoftVoter_" + str(i) + "_best")

all_models.append("stacking")

models_test_accuracies = pd.DataFrame(columns=all_models)
models_train_accuracies = pd.DataFrame(columns=all_models)
models_validation_accuracies = pd.DataFrame(columns=all_models)

average_results = pd.DataFrame(columns=['Model',
                                        'Test Accuracy',
                                        'Test Precision',
                                        'Test Recall',
                                        'Test F-Score',
                                        'AUC',
                                        'TPR',
                                        'FPR',
                                        'TP',
                                        'FP',
                                        'TN',
                                        'FN',
                                        'Train Accuracy',
                                        'Train Precision',
                                        'Train Recall',
                                        'Validation Accuracy',
                                        'Validation Precision',
                                        'Validation Recall',
                                        'PRE',
                                        'REC'
                                        ]
                              )

for i, elem in enumerate(all_models):
    average_results = average_results.append(pd.Series(0, index=average_results.columns), ignore_index=True)
    average_results.ix[i,:]['Model'] = elem

for i in range(0, num_experiments):
    print("\n------------------------------- EXPERIMENT " + str(i+1) + " ------------------------------------")

    seed = i

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, stratify=Y,random_state=seed)

    models_train_accuracies = models_train_accuracies.append(pd.Series(0, index=models_train_accuracies.columns), ignore_index=True)
    models_validation_accuracies = models_validation_accuracies.append(pd.Series(0, index=models_validation_accuracies.columns), ignore_index=True)
    models_test_accuracies = models_test_accuracies.append(pd.Series(0, index=models_test_accuracies.columns), ignore_index=True)

    #Base Models
    for name, model_2 in models:
        average_results,models_test_accuracies, models_train_accuracies, models_validation_accuracies = classify_performance(name, model_2, X_train, X_test, Y_train, Y_test, num_folds, scoring, seed, average_results, models_test_accuracies, models_train_accuracies, models_validation_accuracies)

    #ensemble models
    average_results,models_test_accuracies, models_train_accuracies, models_validation_accuracies  = bagging(average_results, models_test_accuracies, models_train_accuracies, models_validation_accuracies, X_train, X_test, Y_train, Y_test, seed, num_folds, scoring)

    average_results,models_test_accuracies,models_train_accuracies, models_validation_accuracies  = blending(average_results,  models_test_accuracies, models_train_accuracies, models_validation_accuracies, X_train, X_test, Y_train, Y_test, seed, num_folds, scoring)

    average_results,models_test_accuracies,models_train_accuracies, models_validation_accuracies  = soft_voting(average_results,  models_test_accuracies, models_train_accuracies, models_validation_accuracies, X_train, X_test, Y_train, Y_test, seed, num_folds, scoring)

    average_results,models_test_accuracies, models_train_accuracies, models_validation_accuracies  = stacking_function(average_results,  models_test_accuracies, models_train_accuracies, models_validation_accuracies, X_train, X_test, Y_train, Y_test, seed, num_folds, scoring)

```

# Analysis of the results


```python
#calculating the average of the metrics
for key, value in enumerate(all_models):
    for col in average_results.columns:
        if col != 'Model':
            current = average_results.ix[average_results['Model'] == value][col].values
            if col == 'TPR' or col == 'FPR' or col == 'REC' or col == 'PRE':
                division = list(map(truediv, current[0], [num_experiments] *len(current[0])))
            else:
                division = float(current/num_experiments)

            average_results.iloc[average_results.ix[average_results['Model'] == value].index.values.astype(int)[0]][col] = division
```


```python
#sort by the criteria that you want
average_results.sort_values(by=["Test Accuracy", "Test F-Score","Test Precision", "Test Recall", "AUC"], ascending=False, inplace=True)

display(average_results)

for model in all_models:
    line = pd.DataFrame(columns=average_results.columns)

    line = line.append(pd.Series(average_results.iloc[0, :]), ignore_index=True)

    line.to_html('./results/census/' + model + '/average_results.html')
    imgkit.from_file('./results/census/' + model + '/average_results.html', './results/census/' + model + '/average_results.png') # you need to have wkhtmltoimage in your computer  and configure your environmental variable before executing this
    os.remove('./results/census/' + model + '/average_results.html')
```


```python
#confusion matrix and roc curves for each algorithm

for index, row in average_results.iterrows():

    name = average_results.iloc[index, 0]

    tp = average_results.iloc[index, 8]
    fp = average_results.iloc[index, 9]
    tn = average_results.iloc[index, 10]
    fn = average_results.iloc[index, 11]

    roc_auc = average_results.iloc[index, 5]
    tpr = average_results.iloc[index, 6]
    fpr = average_results.iloc[index, 7]
    pre = average_results.iloc[index, 18]
    rec = average_results.iloc[index, 19]

    confusion_matrix = pd.DataFrame({'1': [tp,  fn],
                                     '0': [fp,  tn]
                                    })

    confusion_matrix.rename(index={0:'1',1:'0'}, inplace=True)

    ax = sns.heatmap(confusion_matrix, annot=True,cmap='Blues', fmt='g')


    ax.set(xlabel='Predicted', ylabel='Actual')
    plt.title("Averaged Confusion Matrix of Model " + name)
    plt.savefig('./results/census/' + name + '/confusion_matrix.png')
    plt.show()

    plt.figure(figsize=(20,10))

    plt.plot(fpr, tpr, color = 'blue', label='{} ROC curve (area = {:.2})'.format(name, roc_auc))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.grid()
    plt.savefig('./results/census/' + name + '/roc_curve.png')
    plt.show()

    plt.figure(figsize=(20,10))
    plt.plot(rec, pre, color = 'green', label='Precision-Recall curve of {}'.format(name))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig('./results/census/' + name + '/precision_recall_curve.png')
    plt.show()

```


```python
my_map = {}
for index,value in enumerate(all_models):
    my_map[index] = value

img_name1 = 'all_f_scores'
img_name2 = 'all_accuracies'
dload = os.path.expanduser('~\\Downloads')
save_dir = '.\\results\census'
```


```python
#Plot of the average of accuracies of the different algorithms

copy_accuracy = average_results
copy_accuracy.sort_values(by=["Test Accuracy"], ascending=False, inplace=True)
accuracies_sorted = copy_accuracy["Test Accuracy"]
accuracies_sorted.rename(index = my_map,inplace = True)

cf.go_offline()
init_notebook_mode()


figure = accuracies_sorted.T.iplot(kind = 'bar', asFigure = True, title = 'Average of Accuracies of the Different Algorithms (average of the 30 Experiments)', theme = 'pearl')
iplot(figure,filename=img_name1, image='png')

```


```python
#Plot of the f-scores of the different algorithms

copy_fscores = average_results
copy_fscores.sort_values(by=["Test F-Score"], ascending=False, inplace=True)
fscores_sorted = copy_fscores["Test F-Score"]
fscores_sorted.rename(index = my_map,inplace = True)

cf.go_offline()
init_notebook_mode()


figure = fscores_sorted.T.iplot(kind = 'bar', asFigure = True, title = 'F-Scores of the Different Algorithms (average of the 30 Experiments)', theme = 'solar')
iplot(figure,filename=img_name2, image='png')

```


```python
#roc curves of all algorithms overlapped
copy_roc = average_results
copy_roc.sort_values(by=["TPR"], ascending=False, inplace=True)
tprs_sorted = list(copy_roc["TPR"])
fprs_sorted = list(copy_roc["FPR"])
ticks = list(copy_roc["Model"])

fig1 = plt.figure(figsize=(15,15))
for i in range(0, len(fprs_sorted)):
    plt.plot(fprs_sorted[i], tprs_sorted[i], label='ROC curve of {}'.format(ticks[i]))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves of All Algorithms (average of the 30 Experiments)')
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], 'k--')
plt.grid()
plt.savefig('./results/census/all_roc_curve.png')
```


```python
copy_precisionRecall = average_results
copy_precisionRecall.sort_values(by=["PRE"], ascending=False, inplace=True)
precisions_sorted = list(copy_precisionRecall["PRE"])
recalls_sorted = list(copy_precisionRecall["REC"])
ticks = list(copy_precisionRecall["Model"])

fig1 = plt.figure(figsize=(15,15))
for i in range(0, len(recalls_sorted)):
    plt.plot(recalls_sorted[i], precisions_sorted[i], label='Precision-Recall curve of {}'.format(ticks[i]))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve of All Algorithms (average of the 30 Experiments)')
plt.legend(loc="lower left")
plt.grid()
plt.savefig('./results/census/all_precision_recall.png')
plt.show()
```


```python
# boxplot algorithm comparison
plt.figure(figsize=(20,20))
plt.title('Comparison of the Models Accuracies (average of the 30 Experiments)', fontsize=20)
plt.boxplot(models_test_accuracies.T)
locs, labels=plt.xticks()
x_ticks = []
new_xticks=models_test_accuracies.columns
plt.xticks(locs,new_xticks, rotation=45, horizontalalignment='right')
plt.savefig('./results/census/boxplot_accuracies.png')
plt.show()

```


```python
#saving the plotly images
copyfile('{}\\{}.png'.format(dload, img_name1),
     '{}\\{}.png'.format(save_dir, img_name1))

copyfile('{}\\{}.png'.format(dload, img_name2),
     '{}\\{}.png'.format(save_dir, img_name2))
```

**Finally, we'll analyze the behavior of validation scores vs training scores vs test scores of each algorithm during the 30 experiments in order to check for overfitting/underfitting issues.**


```python
experiments_array = np.arange(1, num_experiments+1, 1)
for model in all_models:
    plt.figure(figsize=(20,10))
    plt.plot(experiments_array, models_train_accuracies[model].T, '*-', color = 'blue',  label = 'Score on Training')
    plt.plot(experiments_array, models_validation_accuracies[model].T, '*-', color = 'yellow', label = 'Score on Validation')
    plt.plot(experiments_array, models_test_accuracies[model].T, 'r--',  color = 'red', label = 'Score on Testing')


    font_size = 15
    plt.xlabel('Experiments', fontsize = font_size)
    plt.ylabel('Accuracy Score', fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.legend(loc = 'best')
    plt.grid()
    plt.suptitle('Training vs Validating vs Testing Scores on ' + model + '', fontsize = 15)

    plt.tight_layout(rect = [0, 0.03, 1, 0.97])

    plt.savefig('./results/census/' + model + '/overfitting_underfitting.png')
    plt.show()
```

# Statistical Tests and Ranking of the Accuracies

- 1) Ranking the results by rows (experiments)
- 2) Friedman test to see if the null hypothesis that the groups of data are similar can be rejected.
- 3) Nemenyi test to see if that difference is statistical significative.


```python
display(models_test_accuracies)
```


```python
#Ranking the results by row (experiment)

models_test_accuracies_transpose = models_test_accuracies.T

models_test_accuracies_ranked = pd.DataFrame(columns=models_test_accuracies.columns)
for i in models_test_accuracies_transpose:
    models_test_accuracies_ranked = models_test_accuracies_ranked.append(models_test_accuracies_transpose.iloc[:,i].pow(-1).rank(method='dense'), ignore_index=True)

temp = models_test_accuracies_ranked.append(models_test_accuracies_ranked.sum(numeric_only=True), ignore_index=True)
temp = temp.sort_values(by=num_experiments, axis=1, ascending=True)

display(temp)
```


```python
# Friedman and nemenyi test
p_values = sp.posthoc_nemenyi_friedman(temp.iloc[0:num_experiments,:])
```


```python
#Assuming that p = 0.05, we can say that...
with open('./results/census/statistic_results.txt', 'w') as f:
    for i in range(0, len(p_values.columns)):
        for j in range(0, len(p_values.columns)):
            name1 = p_values.columns[i]
            name2 = p_values.columns[j]

            if (j >= i):
                break
            if p_values.iloc[i,j] < 0.01:
                if temp.loc[temp.index[num_experiments], name1] < temp.loc[temp.index[num_experiments], name2]:
                    print("Algorithm " + p_values.columns[i] + " is statistically way much better than " + p_values.columns[j], file=f)
                    print("Algorithm " + p_values.columns[i] + " is statistically way much better than " + p_values.columns[j])
                elif temp.loc[temp.index[num_experiments], name1] > temp.loc[temp.index[num_experiments], name2]:
                    print("Algorithm " + p_values.columns[j] + " is statistically way much better than " + p_values.columns[i], file=f)
                    print("Algorithm " + p_values.columns[j] + " is statistically way much better than " + p_values.columns[i])
            elif p_values.iloc[i,j] < 0.05:
                if temp.loc[temp.index[num_experiments], name1] < temp.loc[temp.index[num_experiments], name2]:
                    print("Algorithm " + p_values.columns[i] + " is statistically better than " + p_values.columns[j], file=f)
                    print("Algorithm " + p_values.columns[i] + " is statistically better than " + p_values.columns[j])
                elif temp.loc[temp.index[num_experiments], name1] > temp.loc[temp.index[num_experiments], name2]:
                    print("Algorithm " + p_values.columns[j] + " is statistically better than " + p_values.columns[i], file=f)
                    print("Algorithm " + p_values.columns[j] + " is statistically better than " + p_values.columns[i])
```

# Interpretability/Explainability

## Check Permutation Importance of Encoded Features

Before encoding we can observe directly which feature is more important for each model (independently if the models have already built-in method like the Random Forest or Logistic Regression). We actually did that already right after tunned the parameters for each model.

However, those results can be misleading once we have categorical features that would be later unfolded in extra numerical features through OneHotEncoding in order to improve the accuracy of our model. So, when we build a permutation feature importance table with the encoded features, we obtain more reliable information.


```python
criteria = "Test Accuracy"

for value in every_model:
    print(value[0] + " - Encoded Feature Importances")
    permutation_importance(model, X_test, Y_test)

```

    Random Forest - Encoded Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0064

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 80.01%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0040

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Husband
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 81.28%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0997

                    &plusmn; 0.0046

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-civ-spouse
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 89.24%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0452

                    &plusmn; 0.0036

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.08%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0346

                    &plusmn; 0.0038

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_HS-grad
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.92%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0249

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.76%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0208

                    &plusmn; 0.0027

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Bachelors
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.32%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0138

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age_0.0
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.70%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0122

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Female
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0116

                    &plusmn; 0.0019

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Doctorate
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.90%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0114

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.85%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0078

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.89%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0077

                    &plusmn; 0.0025

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Prof-school
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Male
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Masters
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.10%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0070

                    &plusmn; 0.0023

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Unmarried
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.30%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0063

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-acdm
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.38%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0060

                    &plusmn; 0.0013

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass_Never-worked
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.73%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0049

                    &plusmn; 0.0007

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-voc
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0037

                    &plusmn; 0.0014

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-AF-spouse
            </td>
        </tr>



            <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 87 more &hellip;</i>
                </td>
            </tr>


    </tbody>
</table>






















    Logistic Regression - Encoded Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0064

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 80.01%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0040

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Husband
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 81.28%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0997

                    &plusmn; 0.0046

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-civ-spouse
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 89.24%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0452

                    &plusmn; 0.0036

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.08%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0346

                    &plusmn; 0.0038

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_HS-grad
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.92%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0249

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.76%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0208

                    &plusmn; 0.0027

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Bachelors
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.32%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0138

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age_0.0
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.70%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0122

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Female
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0116

                    &plusmn; 0.0019

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Doctorate
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.90%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0114

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.85%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0078

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.89%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0077

                    &plusmn; 0.0025

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Prof-school
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Male
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Masters
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.10%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0070

                    &plusmn; 0.0023

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Unmarried
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.30%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0063

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-acdm
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.38%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0060

                    &plusmn; 0.0013

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass_Never-worked
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.73%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0049

                    &plusmn; 0.0007

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-voc
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0037

                    &plusmn; 0.0014

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-AF-spouse
            </td>
        </tr>



            <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 87 more &hellip;</i>
                </td>
            </tr>


    </tbody>
</table>






















    MLP - Encoded Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0064

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 80.01%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0040

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Husband
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 81.28%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0997

                    &plusmn; 0.0046

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-civ-spouse
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 89.24%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0452

                    &plusmn; 0.0036

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.08%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0346

                    &plusmn; 0.0038

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_HS-grad
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.92%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0249

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.76%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0208

                    &plusmn; 0.0027

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Bachelors
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.32%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0138

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age_0.0
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.70%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0122

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Female
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0116

                    &plusmn; 0.0019

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Doctorate
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.90%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0114

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.85%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0078

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.89%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0077

                    &plusmn; 0.0025

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Prof-school
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Male
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Masters
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.10%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0070

                    &plusmn; 0.0023

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Unmarried
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.30%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0063

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-acdm
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.38%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0060

                    &plusmn; 0.0013

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass_Never-worked
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.73%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0049

                    &plusmn; 0.0007

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-voc
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0037

                    &plusmn; 0.0014

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-AF-spouse
            </td>
        </tr>



            <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 87 more &hellip;</i>
                </td>
            </tr>


    </tbody>
</table>






















    KNN - Encoded Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0064

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 80.01%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0040

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Husband
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 81.28%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0997

                    &plusmn; 0.0046

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-civ-spouse
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 89.24%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0452

                    &plusmn; 0.0036

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.08%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0346

                    &plusmn; 0.0038

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_HS-grad
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.92%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0249

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.76%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0208

                    &plusmn; 0.0027

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Bachelors
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.32%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0138

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age_0.0
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.70%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0122

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Female
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0116

                    &plusmn; 0.0019

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Doctorate
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.90%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0114

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.85%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0078

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.89%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0077

                    &plusmn; 0.0025

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Prof-school
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Male
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Masters
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.10%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0070

                    &plusmn; 0.0023

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Unmarried
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.30%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0063

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-acdm
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.38%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0060

                    &plusmn; 0.0013

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass_Never-worked
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.73%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0049

                    &plusmn; 0.0007

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-voc
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0037

                    &plusmn; 0.0014

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-AF-spouse
            </td>
        </tr>



            <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 87 more &hellip;</i>
                </td>
            </tr>


    </tbody>
</table>






















    LDA - Encoded Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0064

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 80.01%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0040

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Husband
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 81.28%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0997

                    &plusmn; 0.0046

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-civ-spouse
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 89.24%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0452

                    &plusmn; 0.0036

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.08%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0346

                    &plusmn; 0.0038

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_HS-grad
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.92%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0249

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.76%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0208

                    &plusmn; 0.0027

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Bachelors
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.32%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0138

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age_0.0
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.70%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0122

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Female
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0116

                    &plusmn; 0.0019

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Doctorate
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.90%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0114

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.85%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0078

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.89%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0077

                    &plusmn; 0.0025

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Prof-school
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Male
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Masters
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.10%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0070

                    &plusmn; 0.0023

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Unmarried
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.30%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0063

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-acdm
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.38%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0060

                    &plusmn; 0.0013

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass_Never-worked
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.73%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0049

                    &plusmn; 0.0007

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-voc
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0037

                    &plusmn; 0.0014

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-AF-spouse
            </td>
        </tr>



            <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 87 more &hellip;</i>
                </td>
            </tr>


    </tbody>
</table>






















    DT - Encoded Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0064

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 80.01%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0040

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Husband
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 81.28%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0997

                    &plusmn; 0.0046

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-civ-spouse
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 89.24%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0452

                    &plusmn; 0.0036

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.08%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0346

                    &plusmn; 0.0038

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_HS-grad
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.92%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0249

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.76%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0208

                    &plusmn; 0.0027

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Bachelors
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.32%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0138

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age_0.0
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.70%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0122

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Female
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0116

                    &plusmn; 0.0019

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Doctorate
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.90%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0114

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.85%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0078

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.89%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0077

                    &plusmn; 0.0025

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Prof-school
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Male
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Masters
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.10%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0070

                    &plusmn; 0.0023

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Unmarried
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.30%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0063

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-acdm
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.38%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0060

                    &plusmn; 0.0013

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass_Never-worked
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.73%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0049

                    &plusmn; 0.0007

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-voc
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0037

                    &plusmn; 0.0014

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-AF-spouse
            </td>
        </tr>



            <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 87 more &hellip;</i>
                </td>
            </tr>


    </tbody>
</table>






















    SVC - Encoded Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0064

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 80.01%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0040

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Husband
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 81.28%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0997

                    &plusmn; 0.0046

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-civ-spouse
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 89.24%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0452

                    &plusmn; 0.0036

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.08%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0346

                    &plusmn; 0.0038

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_HS-grad
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.92%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0249

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.76%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0208

                    &plusmn; 0.0027

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Bachelors
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.32%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0138

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age_0.0
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.70%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0122

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Female
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0116

                    &plusmn; 0.0019

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Doctorate
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.90%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0114

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.85%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0078

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.89%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0077

                    &plusmn; 0.0025

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Prof-school
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Male
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Masters
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.10%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0070

                    &plusmn; 0.0023

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Unmarried
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.30%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0063

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-acdm
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.38%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0060

                    &plusmn; 0.0013

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass_Never-worked
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.73%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0049

                    &plusmn; 0.0007

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-voc
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0037

                    &plusmn; 0.0014

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-AF-spouse
            </td>
        </tr>



            <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 87 more &hellip;</i>
                </td>
            </tr>


    </tbody>
</table>






















    SGD - Encoded Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0064

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 80.01%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0040

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Husband
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 81.28%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0997

                    &plusmn; 0.0046

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-civ-spouse
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 89.24%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0452

                    &plusmn; 0.0036

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.08%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0346

                    &plusmn; 0.0038

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_HS-grad
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.92%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0249

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.76%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0208

                    &plusmn; 0.0027

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Bachelors
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.32%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0138

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age_0.0
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.70%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0122

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Female
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0116

                    &plusmn; 0.0019

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Doctorate
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.90%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0114

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.85%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0078

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.89%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0077

                    &plusmn; 0.0025

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Prof-school
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Male
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Masters
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.10%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0070

                    &plusmn; 0.0023

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Unmarried
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.30%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0063

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-acdm
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.38%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0060

                    &plusmn; 0.0013

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass_Never-worked
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.73%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0049

                    &plusmn; 0.0007

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-voc
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0037

                    &plusmn; 0.0014

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-AF-spouse
            </td>
        </tr>



            <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 87 more &hellip;</i>
                </td>
            </tr>


    </tbody>
</table>






















    GraBoost - Encoded Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0064

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 80.01%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0040

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Husband
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 81.28%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0997

                    &plusmn; 0.0046

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-civ-spouse
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 89.24%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0452

                    &plusmn; 0.0036

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.08%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0346

                    &plusmn; 0.0038

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_HS-grad
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.92%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0249

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.76%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0208

                    &plusmn; 0.0027

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Bachelors
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.32%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0138

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age_0.0
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.70%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0122

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Female
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0116

                    &plusmn; 0.0019

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Doctorate
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.90%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0114

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.85%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0078

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.89%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0077

                    &plusmn; 0.0025

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Prof-school
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Male
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Masters
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.10%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0070

                    &plusmn; 0.0023

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Unmarried
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.30%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0063

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-acdm
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.38%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0060

                    &plusmn; 0.0013

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass_Never-worked
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.73%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0049

                    &plusmn; 0.0007

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-voc
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0037

                    &plusmn; 0.0014

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-AF-spouse
            </td>
        </tr>



            <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 87 more &hellip;</i>
                </td>
            </tr>


    </tbody>
</table>






















    AdaBoost - Encoded Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0064

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 80.01%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0040

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Husband
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 81.28%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0997

                    &plusmn; 0.0046

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-civ-spouse
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 89.24%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0452

                    &plusmn; 0.0036

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.08%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0346

                    &plusmn; 0.0038

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_HS-grad
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.92%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0249

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.76%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0208

                    &plusmn; 0.0027

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Bachelors
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.32%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0138

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age_0.0
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.70%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0122

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Female
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0116

                    &plusmn; 0.0019

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Doctorate
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.90%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0114

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.85%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0078

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.89%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0077

                    &plusmn; 0.0025

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Prof-school
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Male
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Masters
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.10%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0070

                    &plusmn; 0.0023

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Unmarried
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.30%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0063

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-acdm
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.38%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0060

                    &plusmn; 0.0013

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass_Never-worked
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.73%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0049

                    &plusmn; 0.0007

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-voc
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0037

                    &plusmn; 0.0014

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-AF-spouse
            </td>
        </tr>



            <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 87 more &hellip;</i>
                </td>
            </tr>


    </tbody>
</table>






















    XGB - Encoded Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0064

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 80.01%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0040

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Husband
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 81.28%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0997

                    &plusmn; 0.0046

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-civ-spouse
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 89.24%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0452

                    &plusmn; 0.0036

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.08%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0346

                    &plusmn; 0.0038

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_HS-grad
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.92%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0249

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.76%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0208

                    &plusmn; 0.0027

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Bachelors
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.32%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0138

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age_0.0
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.70%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0122

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Female
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0116

                    &plusmn; 0.0019

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Doctorate
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.90%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0114

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.85%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0078

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.89%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0077

                    &plusmn; 0.0025

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Prof-school
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Male
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Masters
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.10%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0070

                    &plusmn; 0.0023

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Unmarried
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.30%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0063

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-acdm
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.38%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0060

                    &plusmn; 0.0013

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass_Never-worked
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.73%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0049

                    &plusmn; 0.0007

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-voc
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0037

                    &plusmn; 0.0014

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-AF-spouse
            </td>
        </tr>



            <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 87 more &hellip;</i>
                </td>
            </tr>


    </tbody>
</table>






















    Naive Bayes - Encoded Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0064

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 80.01%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0040

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Husband
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 81.28%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0997

                    &plusmn; 0.0046

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-civ-spouse
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 89.24%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0452

                    &plusmn; 0.0036

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.08%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0346

                    &plusmn; 0.0038

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_HS-grad
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.92%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0249

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.76%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0208

                    &plusmn; 0.0027

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Bachelors
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.32%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0138

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age_0.0
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.70%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0122

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Female
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0116

                    &plusmn; 0.0019

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Doctorate
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.90%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0114

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.85%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0078

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.89%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0077

                    &plusmn; 0.0025

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Prof-school
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Male
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Masters
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.10%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0070

                    &plusmn; 0.0023

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Unmarried
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.30%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0063

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-acdm
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.38%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0060

                    &plusmn; 0.0013

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass_Never-worked
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.73%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0049

                    &plusmn; 0.0007

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-voc
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0037

                    &plusmn; 0.0014

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-AF-spouse
            </td>
        </tr>



            <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 87 more &hellip;</i>
                </td>
            </tr>


    </tbody>
</table>






















    CAT - Encoded Feature Importances




    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>






































        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>

        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0064

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education.num
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 80.01%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1096

                    &plusmn; 0.0040

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Husband
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 81.28%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0997

                    &plusmn; 0.0046

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-civ-spouse
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 89.24%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0452

                    &plusmn; 0.0036

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.gain
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 91.08%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0346

                    &plusmn; 0.0038

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_HS-grad
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 92.92%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0249

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hours.per.week
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 93.76%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0208

                    &plusmn; 0.0027

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Bachelors
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.32%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0138

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                age_0.0
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.70%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0122

                    &plusmn; 0.0022

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Female
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0116

                    &plusmn; 0.0019

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Doctorate
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 95.90%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0114

                    &plusmn; 0.0010

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital.loss
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.85%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0078

                    &plusmn; 0.0028

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                fnlwgt
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 96.89%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0077

                    &plusmn; 0.0025

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Prof-school
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                sex_Male
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0071

                    &plusmn; 0.0008

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Masters
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.10%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0070

                    &plusmn; 0.0023

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                relationship_Unmarried
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.30%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0063

                    &plusmn; 0.0012

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-acdm
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.38%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0060

                    &plusmn; 0.0013

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                workclass_Never-worked
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 97.73%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0049

                    &plusmn; 0.0007

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                education_Assoc-voc
            </td>
        </tr>

        <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0037

                    &plusmn; 0.0014

            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                marital.status_Married-AF-spouse
            </td>
        </tr>



            <tr style="background-color: hsl(120, 100.00%, 98.14%); border: none;">
                <td colspan="2" style="padding: 0 0.5em 0 0.5em; text-align: center; border: none; white-space: nowrap;">
                    <i>&hellip; 87 more &hellip;</i>
                </td>
            </tr>


    </tbody>
</table>






















The values describe the average change in performance of the model when we permute the corresponding feature. Negative values mean that the model accuracy actually improved when the feature values were permuted

We need to be analyze this table as groups of features originated from primordial features, and we have to see which of these groups are predominantly at the top.

We can also compare how the feature importance of these encoded primordial features vary relative to the feature importance of the primordial features.

## Shapley Values
Shapley values are very efficient, but is only available for tree classifiers like Random Forest and Decision Trees.

For an input vector  x , to compute the Shapley value of feature  i , we consider all the possible subset of features that don't include  i , and see how that model prediction would change if we included  i . We then average of all such possible  subsets.

SHAP Values (an acronym from SHapley Additive exPlanations) break down a prediction to show the impact of each feature. Where could you use this?

A model says a bank shouldn't loan someone money, and the bank is legally required to explain the basis for each loan rejection
A healthcare provider wants to identify what factors are driving each patient's risk of some disease so they can directly address those risk factors with targeted health interventions

SHAP values interpret the impact of having a certain value for a given feature in comparison to the prediction we'd make if that feature took some baseline value.


[Source 1](https://www.kaggle.com/dansbecker/shap-values)
[Source 2](https://www.kaggle.com/dansbecker/advanced-uses-of-shap-values)

### Random Forest on Original and Encoded Features


```python
###
X_original_train_1 = X_original_train.iloc[0:500,:]
y_original_train_1 = y_original_train.iloc[0:500]
X_train_1 = X_train.iloc[0:500,:]
Y_train_1 = Y_train.iloc[0:500]
X_test_1 = X_test.iloc[0:500,:]
X_original_test_1 = X_original_test.iloc[0:500,:]
```


```python
clfs_models = []
for index, value in enumerate(tree_classifiers):
    if index % 2 == 0:
        clfs_models.append(value[1].fit(X_original_train_1, y_original_train_1))
    else:
        clfs_models.append(value[1].fit(X_train_1, Y_train_1))
```


```python
#Original Features
explainer = shap.TreeExplainer(clfs_models[0], X_original_train_1, model_output = 'probability')
data_point = X_original_test_1
shap.initjs()
shap_values = explainer.shap_values(data_point)
shap.force_plot(explainer.expected_value[1], shap_values[1],data_point)
```


<div align='center'><img src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAWCAYAAAA1vze2AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAdxJREFUeNq0Vt1Rg0AQJjcpgBJiBWIFkgoMFYhPPAIVECogPuYpdJBYgXQQrMCUkA50V7+d2ZwXuXPGm9khHLu3f9+3l1nkWNvtNqfHLgpfQ1EUS3tz5nAQ0+NIsiAZSc6eDlI8M3J00B/mDuUKDk6kfOebAgW3pkdD0pFcODGW4gKKvOrAUm04MA4QDt1OEIXU9hDigfS5rC1eS5T90gltck1Xrizo257kgySZcNRzgCSxCvgiE9nckPJo2b/B2AcEkk2OwL8bD8gmOKR1GPbaCUqxEgTq0tLvgb6zfo7+DgYGkkWL2tqLDV4RSITfbHPPfJKIrWz4nJQTMPAWA7IbD6imcNaDeDfgk+4No+wZr40BL3g9eQJJCFqRQ54KiSt72lsLpE3o3MCBSxDuq4yOckU2hKXRuwBH3OyMR4g1UpyTYw6mlmBqNdUXRM1NfyF5EPI6JkcpIDBIX8jX6DR/6ckAZJ0wEAdLR8DEk6OfC1Pp8BKo6TQIwPJbvJ6toK5lmuvJoRtfK6Ym1iRYIarRo2UyYHvRN5qpakR3yoizWrouoyuXXQqI185LCw07op5ZyCRGL99h24InP0e9xdQukEKVmhzrqZuRIfwISB//cP3Wk3f8f/yR+BRgAHu00HjLcEQBAAAAAElFTkSuQmCC' /></div><script>!function(t){function e(r){if(n[r])return n[r].exports;var i=n[r]={i:r,l:!1,exports:{}};return t[r].call(i.exports,i,i.exports,e),i.l=!0,i.exports}var n={};e.m=t,e.c=n,e.i=function(t){return t},e.d=function(t,n,r){e.o(t,n)||Object.defineProperty(t,n,{configurable:!1,enumerable:!0,get:r})},e.n=function(t){var n=t&&t.__esModule?function(){return t.default}:function(){return t};return e.d(n,"a",n),n},e.o=function(t,e){return Object.prototype.hasOwnProperty.call(t,e)},e.p="",e(e.s=189)}([function(t,e,n){"use strict";function r(t,e,n,r,o,a,u,c){if(i(e),!t){var s;if(void 0===e)s=new Error("Minified exception occurred; use the non-minified dev environment for the full error message and additional helpful warnings.");else{var l=[n,r,o,a,u,c],f=0;s=new Error(e.replace(/%s/g,function(){return l[f++]})),s.name="Invariant Violation"}throw s.framesToPop=1,s}}var i=function(t){};t.exports=r},function(t,e,n){"use strict";function r(t){for(var e=arguments.length-1,n="Minified React error #"+t+"; visit http://facebook.github.io/react/docs/error-decoder.html?invariant="+t,r=0;r<e;r++)n+="&args[]="+encodeURIComponent(arguments[r+1]);n+=" for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";var i=new Error(n);throw i.name="Invariant Violation",i.framesToPop=1,i}t.exports=r},function(t,e,n){"use strict";var r=n(11),i=r;t.exports=i},function(t,e,n){"use strict";function r(t){if(null===t||void 0===t)throw new TypeError("Object.assign cannot be called with null or undefined");return Object(t)}/*
object-assign
(c) Sindre Sorhus
@license MIT
*/
var i=Object.getOwnPropertySymbols,o=Object.prototype.hasOwnProperty,a=Object.prototype.propertyIsEnumerable;t.exports=function(){try{if(!Object.assign)return!1;var t=new String("abc");if(t[5]="de","5"===Object.getOwnPropertyNames(t)[0])return!1;for(var e={},n=0;n<10;n++)e["_"+String.fromCharCode(n)]=n;if("0123456789"!==Object.getOwnPropertyNames(e).map(function(t){return e[t]}).join(""))return!1;var r={};return"abcdefghijklmnopqrst".split("").forEach(function(t){r[t]=t}),"abcdefghijklmnopqrst"===Object.keys(Object.assign({},r)).join("")}catch(t){return!1}}()?Object.assign:function(t,e){for(var n,u,c=r(t),s=1;s<arguments.length;s++){n=Object(arguments[s]);for(var l in n)o.call(n,l)&&(c[l]=n[l]);if(i){u=i(n);for(var f=0;f<u.length;f++)a.call(n,u[f])&&(c[u[f]]=n[u[f]])}}return c}},function(t,e,n){"use strict";function r(t,e){return 1===t.nodeType&&t.getAttribute(d)===String(e)||8===t.nodeType&&t.nodeValue===" react-text: "+e+" "||8===t.nodeType&&t.nodeValue===" react-empty: "+e+" "}function i(t){for(var e;e=t._renderedComponent;)t=e;return t}function o(t,e){var n=i(t);n._hostNode=e,e[g]=n}function a(t){var e=t._hostNode;e&&(delete e[g],t._hostNode=null)}function u(t,e){if(!(t._flags&v.hasCachedChildNodes)){var n=t._renderedChildren,a=e.firstChild;t:for(var u in n)if(n.hasOwnProperty(u)){var c=n[u],s=i(c)._domID;if(0!==s){for(;null!==a;a=a.nextSibling)if(r(a,s)){o(c,a);continue t}f("32",s)}}t._flags|=v.hasCachedChildNodes}}function c(t){if(t[g])return t[g];for(var e=[];!t[g];){if(e.push(t),!t.parentNode)return null;t=t.parentNode}for(var n,r;t&&(r=t[g]);t=e.pop())n=r,e.length&&u(r,t);return n}function s(t){var e=c(t);return null!=e&&e._hostNode===t?e:null}function l(t){if(void 0===t._hostNode&&f("33"),t._hostNode)return t._hostNode;for(var e=[];!t._hostNode;)e.push(t),t._hostParent||f("34"),t=t._hostParent;for(;e.length;t=e.pop())u(t,t._hostNode);return t._hostNode}var f=n(1),p=n(21),h=n(161),d=(n(0),p.ID_ATTRIBUTE_NAME),v=h,g="__reactInternalInstance$"+Math.random().toString(36).slice(2),m={getClosestInstanceFromNode:c,getInstanceFromNode:s,getNodeFromInstance:l,precacheChildNodes:u,precacheNode:o,uncacheNode:a};t.exports=m},function(t,e,n){"use strict";function r(t,e,n,a){function u(e){return t(e=new Date(+e)),e}return u.floor=u,u.ceil=function(n){return t(n=new Date(n-1)),e(n,1),t(n),n},u.round=function(t){var e=u(t),n=u.ceil(t);return t-e<n-t?e:n},u.offset=function(t,n){return e(t=new Date(+t),null==n?1:Math.floor(n)),t},u.range=function(n,r,i){var o,a=[];if(n=u.ceil(n),i=null==i?1:Math.floor(i),!(n<r&&i>0))return a;do{a.push(o=new Date(+n)),e(n,i),t(n)}while(o<n&&n<r);return a},u.filter=function(n){return r(function(e){if(e>=e)for(;t(e),!n(e);)e.setTime(e-1)},function(t,r){if(t>=t)if(r<0)for(;++r<=0;)for(;e(t,-1),!n(t););else for(;--r>=0;)for(;e(t,1),!n(t););})},n&&(u.count=function(e,r){return i.setTime(+e),o.setTime(+r),t(i),t(o),Math.floor(n(i,o))},u.every=function(t){return t=Math.floor(t),isFinite(t)&&t>0?t>1?u.filter(a?function(e){return a(e)%t==0}:function(e){return u.count(0,e)%t==0}):u:null}),u}e.a=r;var i=new Date,o=new Date},function(t,e,n){"use strict";var r=!("undefined"==typeof window||!window.document||!window.document.createElement),i={canUseDOM:r,canUseWorkers:"undefined"!=typeof Worker,canUseEventListeners:r&&!(!window.addEventListener&&!window.attachEvent),canUseViewport:r&&!!window.screen,isInWorker:!r};t.exports=i},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(101);n.d(e,"bisect",function(){return r.a}),n.d(e,"bisectRight",function(){return r.b}),n.d(e,"bisectLeft",function(){return r.c});var i=n(19);n.d(e,"ascending",function(){return i.a});var o=n(102);n.d(e,"bisector",function(){return o.a});var a=n(193);n.d(e,"cross",function(){return a.a});var u=n(194);n.d(e,"descending",function(){return u.a});var c=n(103);n.d(e,"deviation",function(){return c.a});var s=n(104);n.d(e,"extent",function(){return s.a});var l=n(195);n.d(e,"histogram",function(){return l.a});var f=n(205);n.d(e,"thresholdFreedmanDiaconis",function(){return f.a});var p=n(206);n.d(e,"thresholdScott",function(){return p.a});var h=n(108);n.d(e,"thresholdSturges",function(){return h.a});var d=n(197);n.d(e,"max",function(){return d.a});var v=n(198);n.d(e,"mean",function(){return v.a});var g=n(199);n.d(e,"median",function(){return g.a});var m=n(200);n.d(e,"merge",function(){return m.a});var y=n(105);n.d(e,"min",function(){return y.a});var _=n(106);n.d(e,"pairs",function(){return _.a});var b=n(201);n.d(e,"permute",function(){return b.a});var x=n(59);n.d(e,"quantile",function(){return x.a});var w=n(107);n.d(e,"range",function(){return w.a});var C=n(202);n.d(e,"scan",function(){return C.a});var k=n(203);n.d(e,"shuffle",function(){return k.a});var E=n(204);n.d(e,"sum",function(){return E.a});var M=n(109);n.d(e,"ticks",function(){return M.a}),n.d(e,"tickIncrement",function(){return M.b}),n.d(e,"tickStep",function(){return M.c});var T=n(110);n.d(e,"transpose",function(){return T.a});var S=n(111);n.d(e,"variance",function(){return S.a});var N=n(207);n.d(e,"zip",function(){return N.a})},function(t,e,n){"use strict";function r(t,e){this._groups=t,this._parents=e}function i(){return new r([[document.documentElement]],R)}n.d(e,"c",function(){return R}),e.b=r;var o=n(283),a=n(284),u=n(272),c=n(266),s=n(132),l=n(271),f=n(276),p=n(279),h=n(286),d=n(263),v=n(278),g=n(277),m=n(285),y=n(270),_=n(269),b=n(262),x=n(134),w=n(280),C=n(264),k=n(287),E=n(273),M=n(281),T=n(275),S=n(261),N=n(274),A=n(282),P=n(265),O=n(267),I=n(70),D=n(268),R=[null];r.prototype=i.prototype={constructor:r,select:o.a,selectAll:a.a,filter:u.a,data:c.a,enter:s.a,exit:l.a,merge:f.a,order:p.a,sort:h.a,call:d.a,nodes:v.a,node:g.a,size:m.a,empty:y.a,each:_.a,attr:b.a,style:x.b,property:w.a,classed:C.a,text:k.a,html:E.a,raise:M.a,lower:T.a,append:S.a,insert:N.a,remove:A.a,clone:P.a,datum:O.a,on:I.c,dispatch:D.a},e.a=i},function(t,e,n){"use strict";var r=null;t.exports={debugTool:r}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(61);n.d(e,"color",function(){return r.a}),n.d(e,"rgb",function(){return r.b}),n.d(e,"hsl",function(){return r.c});var i=n(218);n.d(e,"lab",function(){return i.a}),n.d(e,"hcl",function(){return i.b});var o=n(217);n.d(e,"cubehelix",function(){return o.a})},function(t,e,n){"use strict";function r(t){return function(){return t}}var i=function(){};i.thatReturns=r,i.thatReturnsFalse=r(!1),i.thatReturnsTrue=r(!0),i.thatReturnsNull=r(null),i.thatReturnsThis=function(){return this},i.thatReturnsArgument=function(t){return t},t.exports=i},function(t,e,n){"use strict";function r(){S.ReactReconcileTransaction&&w||l("123")}function i(){this.reinitializeTransaction(),this.dirtyComponentsLength=null,this.callbackQueue=p.getPooled(),this.reconcileTransaction=S.ReactReconcileTransaction.getPooled(!0)}function o(t,e,n,i,o,a){return r(),w.batchedUpdates(t,e,n,i,o,a)}function a(t,e){return t._mountOrder-e._mountOrder}function u(t){var e=t.dirtyComponentsLength;e!==y.length&&l("124",e,y.length),y.sort(a),_++;for(var n=0;n<e;n++){var r=y[n],i=r._pendingCallbacks;r._pendingCallbacks=null;var o;if(d.logTopLevelRenders){var u=r;r._currentElement.type.isReactTopLevelWrapper&&(u=r._renderedComponent),o="React update: "+u.getName(),console.time(o)}if(v.performUpdateIfNecessary(r,t.reconcileTransaction,_),o&&console.timeEnd(o),i)for(var c=0;c<i.length;c++)t.callbackQueue.enqueue(i[c],r.getPublicInstance())}}function c(t){if(r(),!w.isBatchingUpdates)return void w.batchedUpdates(c,t);y.push(t),null==t._updateBatchNumber&&(t._updateBatchNumber=_+1)}function s(t,e){m(w.isBatchingUpdates,"ReactUpdates.asap: Can't enqueue an asap callback in a context whereupdates are not being batched."),b.enqueue(t,e),x=!0}var l=n(1),f=n(3),p=n(159),h=n(18),d=n(164),v=n(24),g=n(55),m=n(0),y=[],_=0,b=p.getPooled(),x=!1,w=null,C={initialize:function(){this.dirtyComponentsLength=y.length},close:function(){this.dirtyComponentsLength!==y.length?(y.splice(0,this.dirtyComponentsLength),M()):y.length=0}},k={initialize:function(){this.callbackQueue.reset()},close:function(){this.callbackQueue.notifyAll()}},E=[C,k];f(i.prototype,g,{getTransactionWrappers:function(){return E},destructor:function(){this.dirtyComponentsLength=null,p.release(this.callbackQueue),this.callbackQueue=null,S.ReactReconcileTransaction.release(this.reconcileTransaction),this.reconcileTransaction=null},perform:function(t,e,n){return g.perform.call(this,this.reconcileTransaction.perform,this.reconcileTransaction,t,e,n)}}),h.addPoolingTo(i);var M=function(){for(;y.length||x;){if(y.length){var t=i.getPooled();t.perform(u,null,t),i.release(t)}if(x){x=!1;var e=b;b=p.getPooled(),e.notifyAll(),p.release(e)}}},T={injectReconcileTransaction:function(t){t||l("126"),S.ReactReconcileTransaction=t},injectBatchingStrategy:function(t){t||l("127"),"function"!=typeof t.batchedUpdates&&l("128"),"boolean"!=typeof t.isBatchingUpdates&&l("129"),w=t}},S={ReactReconcileTransaction:null,batchedUpdates:o,enqueueUpdate:c,flushBatchedUpdates:M,injection:T,asap:s};t.exports=S},function(t,e,n){"use strict";n.d(e,"e",function(){return r}),n.d(e,"d",function(){return i}),n.d(e,"c",function(){return o}),n.d(e,"b",function(){return a}),n.d(e,"a",function(){return u});var r=1e3,i=6e4,o=36e5,a=864e5,u=6048e5},function(t,e,n){"use strict";function r(t,e,n,r){this.dispatchConfig=t,this._targetInst=e,this.nativeEvent=n;var i=this.constructor.Interface;for(var o in i)if(i.hasOwnProperty(o)){var u=i[o];u?this[o]=u(n):"target"===o?this.target=r:this[o]=n[o]}var c=null!=n.defaultPrevented?n.defaultPrevented:!1===n.returnValue;return this.isDefaultPrevented=c?a.thatReturnsTrue:a.thatReturnsFalse,this.isPropagationStopped=a.thatReturnsFalse,this}var i=n(3),o=n(18),a=n(11),u=(n(2),["dispatchConfig","_targetInst","nativeEvent","isDefaultPrevented","isPropagationStopped","_dispatchListeners","_dispatchInstances"]),c={type:null,target:null,currentTarget:a.thatReturnsNull,eventPhase:null,bubbles:null,cancelable:null,timeStamp:function(t){return t.timeStamp||Date.now()},defaultPrevented:null,isTrusted:null};i(r.prototype,{preventDefault:function(){this.defaultPrevented=!0;var t=this.nativeEvent;t&&(t.preventDefault?t.preventDefault():"unknown"!=typeof t.returnValue&&(t.returnValue=!1),this.isDefaultPrevented=a.thatReturnsTrue)},stopPropagation:function(){var t=this.nativeEvent;t&&(t.stopPropagation?t.stopPropagation():"unknown"!=typeof t.cancelBubble&&(t.cancelBubble=!0),this.isPropagationStopped=a.thatReturnsTrue)},persist:function(){this.isPersistent=a.thatReturnsTrue},isPersistent:a.thatReturnsFalse,destructor:function(){var t=this.constructor.Interface;for(var e in t)this[e]=null;for(var n=0;n<u.length;n++)this[u[n]]=null}}),r.Interface=c,r.augmentClass=function(t,e){var n=this,r=function(){};r.prototype=n.prototype;var a=new r;i(a,t.prototype),t.prototype=a,t.prototype.constructor=t,t.Interface=i({},n.Interface,e),t.augmentClass=n.augmentClass,o.addPoolingTo(t,o.fourArgumentPooler)},o.addPoolingTo(r,o.fourArgumentPooler),t.exports=r},function(t,e,n){"use strict";var r={current:null};t.exports=r},function(t,e,n){"use strict";n.d(e,"a",function(){return i}),n.d(e,"b",function(){return o});var r=Array.prototype,i=r.map,o=r.slice},function(t,e,n){"use strict";e.a=function(t){return function(){return t}}},function(t,e,n){"use strict";var r=n(1),i=(n(0),function(t){var e=this;if(e.instancePool.length){var n=e.instancePool.pop();return e.call(n,t),n}return new e(t)}),o=function(t,e){var n=this;if(n.instancePool.length){var r=n.instancePool.pop();return n.call(r,t,e),r}return new n(t,e)},a=function(t,e,n){var r=this;if(r.instancePool.length){var i=r.instancePool.pop();return r.call(i,t,e,n),i}return new r(t,e,n)},u=function(t,e,n,r){var i=this;if(i.instancePool.length){var o=i.instancePool.pop();return i.call(o,t,e,n,r),o}return new i(t,e,n,r)},c=function(t){var e=this;t instanceof e||r("25"),t.destructor(),e.instancePool.length<e.poolSize&&e.instancePool.push(t)},s=i,l=function(t,e){var n=t;return n.instancePool=[],n.getPooled=e||s,n.poolSize||(n.poolSize=10),n.release=c,n},f={addPoolingTo:l,oneArgumentPooler:i,twoArgumentPooler:o,threeArgumentPooler:a,fourArgumentPooler:u};t.exports=f},function(t,e,n){"use strict";e.a=function(t,e){return t<e?-1:t>e?1:t>=e?0:NaN}},function(t,e,n){"use strict";function r(t){if(d){var e=t.node,n=t.children;if(n.length)for(var r=0;r<n.length;r++)v(e,n[r],null);else null!=t.html?f(e,t.html):null!=t.text&&h(e,t.text)}}function i(t,e){t.parentNode.replaceChild(e.node,t),r(e)}function o(t,e){d?t.children.push(e):t.node.appendChild(e.node)}function a(t,e){d?t.html=e:f(t.node,e)}function u(t,e){d?t.text=e:h(t.node,e)}function c(){return this.node.nodeName}function s(t){return{node:t,children:[],html:null,text:null,toString:c}}var l=n(83),f=n(57),p=n(91),h=n(176),d="undefined"!=typeof document&&"number"==typeof document.documentMode||"undefined"!=typeof navigator&&"string"==typeof navigator.userAgent&&/\bEdge\/\d/.test(navigator.userAgent),v=p(function(t,e,n){11===e.node.nodeType||1===e.node.nodeType&&"object"===e.node.nodeName.toLowerCase()&&(null==e.node.namespaceURI||e.node.namespaceURI===l.html)?(r(e),t.insertBefore(e.node,n)):(t.insertBefore(e.node,n),r(e))});s.insertTreeBefore=v,s.replaceChildWithTree=i,s.queueChild=o,s.queueHTML=a,s.queueText=u,t.exports=s},function(t,e,n){"use strict";function r(t,e){return(t&e)===e}var i=n(1),o=(n(0),{MUST_USE_PROPERTY:1,HAS_BOOLEAN_VALUE:4,HAS_NUMERIC_VALUE:8,HAS_POSITIVE_NUMERIC_VALUE:24,HAS_OVERLOADED_BOOLEAN_VALUE:32,injectDOMPropertyConfig:function(t){var e=o,n=t.Properties||{},a=t.DOMAttributeNamespaces||{},c=t.DOMAttributeNames||{},s=t.DOMPropertyNames||{},l=t.DOMMutationMethods||{};t.isCustomAttribute&&u._isCustomAttributeFunctions.push(t.isCustomAttribute);for(var f in n){u.properties.hasOwnProperty(f)&&i("48",f);var p=f.toLowerCase(),h=n[f],d={attributeName:p,attributeNamespace:null,propertyName:f,mutationMethod:null,mustUseProperty:r(h,e.MUST_USE_PROPERTY),hasBooleanValue:r(h,e.HAS_BOOLEAN_VALUE),hasNumericValue:r(h,e.HAS_NUMERIC_VALUE),hasPositiveNumericValue:r(h,e.HAS_POSITIVE_NUMERIC_VALUE),hasOverloadedBooleanValue:r(h,e.HAS_OVERLOADED_BOOLEAN_VALUE)};if(d.hasBooleanValue+d.hasNumericValue+d.hasOverloadedBooleanValue<=1||i("50",f),c.hasOwnProperty(f)){var v=c[f];d.attributeName=v}a.hasOwnProperty(f)&&(d.attributeNamespace=a[f]),s.hasOwnProperty(f)&&(d.propertyName=s[f]),l.hasOwnProperty(f)&&(d.mutationMethod=l[f]),u.properties[f]=d}}}),a=":A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD",u={ID_ATTRIBUTE_NAME:"data-reactid",ROOT_ATTRIBUTE_NAME:"data-reactroot",ATTRIBUTE_NAME_START_CHAR:a,ATTRIBUTE_NAME_CHAR:a+"\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040",properties:{},getPossibleStandardName:null,_isCustomAttributeFunctions:[],isCustomAttribute:function(t){for(var e=0;e<u._isCustomAttributeFunctions.length;e++){if((0,u._isCustomAttributeFunctions[e])(t))return!0}return!1},injection:o};t.exports=u},function(t,e,n){"use strict";function r(t){return"button"===t||"input"===t||"select"===t||"textarea"===t}function i(t,e,n){switch(t){case"onClick":case"onClickCapture":case"onDoubleClick":case"onDoubleClickCapture":case"onMouseDown":case"onMouseDownCapture":case"onMouseMove":case"onMouseMoveCapture":case"onMouseUp":case"onMouseUpCapture":return!(!n.disabled||!r(e));default:return!1}}var o=n(1),a=n(84),u=n(52),c=n(88),s=n(169),l=n(170),f=(n(0),{}),p=null,h=function(t,e){t&&(u.executeDispatchesInOrder(t,e),t.isPersistent()||t.constructor.release(t))},d=function(t){return h(t,!0)},v=function(t){return h(t,!1)},g=function(t){return"."+t._rootNodeID},m={injection:{injectEventPluginOrder:a.injectEventPluginOrder,injectEventPluginsByName:a.injectEventPluginsByName},putListener:function(t,e,n){"function"!=typeof n&&o("94",e,typeof n);var r=g(t);(f[e]||(f[e]={}))[r]=n;var i=a.registrationNameModules[e];i&&i.didPutListener&&i.didPutListener(t,e,n)},getListener:function(t,e){var n=f[e];if(i(e,t._currentElement.type,t._currentElement.props))return null;var r=g(t);return n&&n[r]},deleteListener:function(t,e){var n=a.registrationNameModules[e];n&&n.willDeleteListener&&n.willDeleteListener(t,e);var r=f[e];if(r){delete r[g(t)]}},deleteAllListeners:function(t){var e=g(t);for(var n in f)if(f.hasOwnProperty(n)&&f[n][e]){var r=a.registrationNameModules[n];r&&r.willDeleteListener&&r.willDeleteListener(t,n),delete f[n][e]}},extractEvents:function(t,e,n,r){for(var i,o=a.plugins,u=0;u<o.length;u++){var c=o[u];if(c){var l=c.extractEvents(t,e,n,r);l&&(i=s(i,l))}}return i},enqueueEvents:function(t){t&&(p=s(p,t))},processEventQueue:function(t){var e=p;p=null,t?l(e,d):l(e,v),p&&o("95"),c.rethrowCaughtError()},__purge:function(){f={}},__getListenerBank:function(){return f}};t.exports=m},function(t,e,n){"use strict";function r(t,e,n){var r=e.dispatchConfig.phasedRegistrationNames[n];return m(t,r)}function i(t,e,n){var i=r(t,n,e);i&&(n._dispatchListeners=v(n._dispatchListeners,i),n._dispatchInstances=v(n._dispatchInstances,t))}function o(t){t&&t.dispatchConfig.phasedRegistrationNames&&d.traverseTwoPhase(t._targetInst,i,t)}function a(t){if(t&&t.dispatchConfig.phasedRegistrationNames){var e=t._targetInst,n=e?d.getParentInstance(e):null;d.traverseTwoPhase(n,i,t)}}function u(t,e,n){if(n&&n.dispatchConfig.registrationName){var r=n.dispatchConfig.registrationName,i=m(t,r);i&&(n._dispatchListeners=v(n._dispatchListeners,i),n._dispatchInstances=v(n._dispatchInstances,t))}}function c(t){t&&t.dispatchConfig.registrationName&&u(t._targetInst,null,t)}function s(t){g(t,o)}function l(t){g(t,a)}function f(t,e,n,r){d.traverseEnterLeave(n,r,u,t,e)}function p(t){g(t,c)}var h=n(22),d=n(52),v=n(169),g=n(170),m=(n(2),h.getListener),y={accumulateTwoPhaseDispatches:s,accumulateTwoPhaseDispatchesSkipTarget:l,accumulateDirectDispatches:p,accumulateEnterLeaveDispatches:f};t.exports=y},function(t,e,n){"use strict";function r(){i.attachRefs(this,this._currentElement)}var i=n(382),o=(n(9),n(2),{mountComponent:function(t,e,n,i,o,a){var u=t.mountComponent(e,n,i,o,a);return t._currentElement&&null!=t._currentElement.ref&&e.getReactMountReady().enqueue(r,t),u},getHostNode:function(t){return t.getHostNode()},unmountComponent:function(t,e){i.detachRefs(t,t._currentElement),t.unmountComponent(e)},receiveComponent:function(t,e,n,o){var a=t._currentElement;if(e!==a||o!==t._context){var u=i.shouldUpdateRefs(a,e);u&&i.detachRefs(t,a),t.receiveComponent(e,n,o),u&&t._currentElement&&null!=t._currentElement.ref&&n.getReactMountReady().enqueue(r,t)}},performUpdateIfNecessary:function(t,e,n){t._updateBatchNumber===n&&t.performUpdateIfNecessary(e)}});t.exports=o},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(14),o=n(94),a={view:function(t){if(t.view)return t.view;var e=o(t);if(e.window===e)return e;var n=e.ownerDocument;return n?n.defaultView||n.parentWindow:window},detail:function(t){return t.detail||0}};i.augmentClass(r,a),t.exports=r},function(t,e,n){"use strict";var r=n(3),i=n(178),o=n(414),a=n(415),u=n(27),c=n(416),s=n(417),l=n(418),f=n(422),p=u.createElement,h=u.createFactory,d=u.cloneElement,v=r,g=function(t){return t},m={Children:{map:o.map,forEach:o.forEach,count:o.count,toArray:o.toArray,only:f},Component:i.Component,PureComponent:i.PureComponent,createElement:p,cloneElement:d,isValidElement:u.isValidElement,PropTypes:c,createClass:l,createFactory:h,createMixin:g,DOM:a,version:s,__spread:v};t.exports=m},function(t,e,n){"use strict";function r(t){return void 0!==t.ref}function i(t){return void 0!==t.key}var o=n(3),a=n(15),u=(n(2),n(182),Object.prototype.hasOwnProperty),c=n(180),s={key:!0,ref:!0,__self:!0,__source:!0},l=function(t,e,n,r,i,o,a){var u={$$typeof:c,type:t,key:e,ref:n,props:a,_owner:o};return u};l.createElement=function(t,e,n){var o,c={},f=null,p=null;if(null!=e){r(e)&&(p=e.ref),i(e)&&(f=""+e.key),void 0===e.__self?null:e.__self,void 0===e.__source?null:e.__source;for(o in e)u.call(e,o)&&!s.hasOwnProperty(o)&&(c[o]=e[o])}var h=arguments.length-2;if(1===h)c.children=n;else if(h>1){for(var d=Array(h),v=0;v<h;v++)d[v]=arguments[v+2];c.children=d}if(t&&t.defaultProps){var g=t.defaultProps;for(o in g)void 0===c[o]&&(c[o]=g[o])}return l(t,f,p,0,0,a.current,c)},l.createFactory=function(t){var e=l.createElement.bind(null,t);return e.type=t,e},l.cloneAndReplaceKey=function(t,e){return l(t.type,e,t.ref,t._self,t._source,t._owner,t.props)},l.cloneElement=function(t,e,n){var c,f=o({},t.props),p=t.key,h=t.ref,d=(t._self,t._source,t._owner);if(null!=e){r(e)&&(h=e.ref,d=a.current),i(e)&&(p=""+e.key);var v;t.type&&t.type.defaultProps&&(v=t.type.defaultProps);for(c in e)u.call(e,c)&&!s.hasOwnProperty(c)&&(void 0===e[c]&&void 0!==v?f[c]=v[c]:f[c]=e[c])}var g=arguments.length-2;if(1===g)f.children=n;else if(g>1){for(var m=Array(g),y=0;y<g;y++)m[y]=arguments[y+2];f.children=m}return l(t.type,p,h,0,0,d,f)},l.isValidElement=function(t){return"object"==typeof t&&null!==t&&t.$$typeof===c},t.exports=l},function(t,e,n){"use strict";e.a=function(t){return null===t?NaN:+t}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(219);n.d(e,"formatDefaultLocale",function(){return r.a}),n.d(e,"format",function(){return r.b}),n.d(e,"formatPrefix",function(){return r.c});var i=n(117);n.d(e,"formatLocale",function(){return i.a});var o=n(115);n.d(e,"formatSpecifier",function(){return o.a});var a=n(225);n.d(e,"precisionFixed",function(){return a.a});var u=n(226);n.d(e,"precisionPrefix",function(){return u.a});var c=n(227);n.d(e,"precisionRound",function(){return c.a})},function(t,e,n){"use strict";var r=n(65);n.d(e,"b",function(){return r.a});var i=(n(118),n(64),n(119),n(121),n(43));n.d(e,"a",function(){return i.a});var o=(n(122),n(233));n.d(e,"c",function(){return o.a});var a=(n(124),n(235),n(237),n(123),n(230),n(231),n(229),n(228));n.d(e,"d",function(){return a.a});n(232)},function(t,e,n){"use strict";function r(t,e){return function(n){return t+n*e}}function i(t,e,n){return t=Math.pow(t,n),e=Math.pow(e,n)-t,n=1/n,function(r){return Math.pow(t+r*e,n)}}function o(t,e){var i=e-t;return i?r(t,i>180||i<-180?i-360*Math.round(i/360):i):n.i(c.a)(isNaN(t)?e:t)}function a(t){return 1==(t=+t)?u:function(e,r){return r-e?i(e,r,t):n.i(c.a)(isNaN(e)?r:e)}}function u(t,e){var i=e-t;return i?r(t,i):n.i(c.a)(isNaN(t)?e:t)}e.b=o,e.c=a,e.a=u;var c=n(120)},function(t,e,n){"use strict";var r=n(238);n.d(e,"a",function(){return r.a})},function(t,e,n){"use strict";e.a=function(t){return t.match(/.{6}/g).map(function(t){return"#"+t})}},function(t,e,n){"use strict";function r(t){var e=t.domain;return t.ticks=function(t){var r=e();return n.i(o.ticks)(r[0],r[r.length-1],null==t?10:t)},t.tickFormat=function(t,r){return n.i(c.a)(e(),t,r)},t.nice=function(r){null==r&&(r=10);var i,a=e(),u=0,c=a.length-1,s=a[u],l=a[c];return l<s&&(i=s,s=l,l=i,i=u,u=c,c=i),i=n.i(o.tickIncrement)(s,l,r),i>0?(s=Math.floor(s/i)*i,l=Math.ceil(l/i)*i,i=n.i(o.tickIncrement)(s,l,r)):i<0&&(s=Math.ceil(s*i)/i,l=Math.floor(l*i)/i,i=n.i(o.tickIncrement)(s,l,r)),i>0?(a[u]=Math.floor(s/i)*i,a[c]=Math.ceil(l/i)*i,e(a)):i<0&&(a[u]=Math.ceil(s*i)/i,a[c]=Math.floor(l*i)/i,e(a)),t},t}function i(){var t=n.i(u.a)(u.b,a.a);return t.copy=function(){return n.i(u.c)(t,i())},r(t)}e.b=r,e.a=i;var o=n(7),a=n(30),u=n(44),c=n(253)},function(t,e,n){"use strict";function r(t){return t>1?0:t<-1?h:Math.acos(t)}function i(t){return t>=1?d:t<=-1?-d:Math.asin(t)}n.d(e,"g",function(){return o}),n.d(e,"m",function(){return a}),n.d(e,"h",function(){return u}),n.d(e,"e",function(){return c}),n.d(e,"j",function(){return s}),n.d(e,"i",function(){return l}),n.d(e,"d",function(){return f}),n.d(e,"a",function(){return p}),n.d(e,"b",function(){return h}),n.d(e,"f",function(){return d}),n.d(e,"c",function(){return v}),e.l=r,e.k=i;var o=Math.abs,a=Math.atan2,u=Math.cos,c=Math.max,s=Math.min,l=Math.sin,f=Math.sqrt,p=1e-12,h=Math.PI,d=h/2,v=2*h},function(t,e,n){"use strict";e.a=function(t,e){if((i=t.length)>1)for(var n,r,i,o=1,a=t[e[0]],u=a.length;o<i;++o)for(r=a,a=t[e[o]],n=0;n<u;++n)a[n][1]+=a[n][0]=isNaN(r[n][1])?r[n][0]:r[n][1]}},function(t,e,n){"use strict";e.a=function(t){for(var e=t.length,n=new Array(e);--e>=0;)n[e]=e;return n}},function(t,e,n){(function(t,r){var i;(function(){function o(t,e,n){switch(n.length){case 0:return t.call(e);case 1:return t.call(e,n[0]);case 2:return t.call(e,n[0],n[1]);case 3:return t.call(e,n[0],n[1],n[2])}return t.apply(e,n)}function a(t,e,n,r){for(var i=-1,o=null==t?0:t.length;++i<o;){var a=t[i];e(r,a,n(a),t)}return r}function u(t,e){for(var n=-1,r=null==t?0:t.length;++n<r&&!1!==e(t[n],n,t););return t}function c(t,e){for(var n=null==t?0:t.length;n--&&!1!==e(t[n],n,t););return t}function s(t,e){for(var n=-1,r=null==t?0:t.length;++n<r;)if(!e(t[n],n,t))return!1;return!0}function l(t,e){for(var n=-1,r=null==t?0:t.length,i=0,o=[];++n<r;){var a=t[n];e(a,n,t)&&(o[i++]=a)}return o}function f(t,e){return!!(null==t?0:t.length)&&w(t,e,0)>-1}function p(t,e,n){for(var r=-1,i=null==t?0:t.length;++r<i;)if(n(e,t[r]))return!0;return!1}function h(t,e){for(var n=-1,r=null==t?0:t.length,i=Array(r);++n<r;)i[n]=e(t[n],n,t);return i}function d(t,e){for(var n=-1,r=e.length,i=t.length;++n<r;)t[i+n]=e[n];return t}function v(t,e,n,r){var i=-1,o=null==t?0:t.length;for(r&&o&&(n=t[++i]);++i<o;)n=e(n,t[i],i,t);return n}function g(t,e,n,r){var i=null==t?0:t.length;for(r&&i&&(n=t[--i]);i--;)n=e(n,t[i],i,t);return n}function m(t,e){for(var n=-1,r=null==t?0:t.length;++n<r;)if(e(t[n],n,t))return!0;return!1}function y(t){return t.split("")}function _(t){return t.match(Ue)||[]}function b(t,e,n){var r;return n(t,function(t,n,i){if(e(t,n,i))return r=n,!1}),r}function x(t,e,n,r){for(var i=t.length,o=n+(r?1:-1);r?o--:++o<i;)if(e(t[o],o,t))return o;return-1}function w(t,e,n){return e===e?$(t,e,n):x(t,k,n)}function C(t,e,n,r){for(var i=n-1,o=t.length;++i<o;)if(r(t[i],e))return i;return-1}function k(t){return t!==t}function E(t,e){var n=null==t?0:t.length;return n?A(t,e)/n:It}function M(t){return function(e){return null==e?nt:e[t]}}function T(t){return function(e){return null==t?nt:t[e]}}function S(t,e,n,r,i){return i(t,function(t,i,o){n=r?(r=!1,t):e(n,t,i,o)}),n}function N(t,e){var n=t.length;for(t.sort(e);n--;)t[n]=t[n].value;return t}function A(t,e){for(var n,r=-1,i=t.length;++r<i;){var o=e(t[r]);o!==nt&&(n=n===nt?o:n+o)}return n}function P(t,e){for(var n=-1,r=Array(t);++n<t;)r[n]=e(n);return r}function O(t,e){return h(e,function(e){return[e,t[e]]})}function I(t){return function(e){return t(e)}}function D(t,e){return h(e,function(e){return t[e]})}function R(t,e){return t.has(e)}function L(t,e){for(var n=-1,r=t.length;++n<r&&w(e,t[n],0)>-1;);return n}function U(t,e){for(var n=t.length;n--&&w(e,t[n],0)>-1;);return n}function F(t,e){for(var n=t.length,r=0;n--;)t[n]===e&&++r;return r}function j(t){return"\\"+En[t]}function B(t,e){return null==t?nt:t[e]}function V(t){return gn.test(t)}function W(t){return mn.test(t)}function z(t){for(var e,n=[];!(e=t.next()).done;)n.push(e.value);return n}function H(t){var e=-1,n=Array(t.size);return t.forEach(function(t,r){n[++e]=[r,t]}),n}function q(t,e){return function(n){return t(e(n))}}function Y(t,e){for(var n=-1,r=t.length,i=0,o=[];++n<r;){var a=t[n];a!==e&&a!==ct||(t[n]=ct,o[i++]=n)}return o}function K(t){var e=-1,n=Array(t.size);return t.forEach(function(t){n[++e]=t}),n}function G(t){var e=-1,n=Array(t.size);return t.forEach(function(t){n[++e]=[t,t]}),n}function $(t,e,n){for(var r=n-1,i=t.length;++r<i;)if(t[r]===e)return r;return-1}function X(t,e,n){for(var r=n+1;r--;)if(t[r]===e)return r;return r}function Q(t){return V(t)?J(t):Wn(t)}function Z(t){return V(t)?tt(t):y(t)}function J(t){for(var e=dn.lastIndex=0;dn.test(t);)++e;return e}function tt(t){return t.match(dn)||[]}function et(t){return t.match(vn)||[]}var nt,rt=200,it="Unsupported core-js use. Try https://npms.io/search?q=ponyfill.",ot="Expected a function",at="__lodash_hash_undefined__",ut=500,ct="__lodash_placeholder__",st=1,lt=2,ft=4,pt=1,ht=2,dt=1,vt=2,gt=4,mt=8,yt=16,_t=32,bt=64,xt=128,wt=256,Ct=512,kt=30,Et="...",Mt=800,Tt=16,St=1,Nt=2,At=1/0,Pt=9007199254740991,Ot=1.7976931348623157e308,It=NaN,Dt=4294967295,Rt=Dt-1,Lt=Dt>>>1,Ut=[["ary",xt],["bind",dt],["bindKey",vt],["curry",mt],["curryRight",yt],["flip",Ct],["partial",_t],["partialRight",bt],["rearg",wt]],Ft="[object Arguments]",jt="[object Array]",Bt="[object AsyncFunction]",Vt="[object Boolean]",Wt="[object Date]",zt="[object DOMException]",Ht="[object Error]",qt="[object Function]",Yt="[object GeneratorFunction]",Kt="[object Map]",Gt="[object Number]",$t="[object Null]",Xt="[object Object]",Qt="[object Proxy]",Zt="[object RegExp]",Jt="[object Set]",te="[object String]",ee="[object Symbol]",ne="[object Undefined]",re="[object WeakMap]",ie="[object WeakSet]",oe="[object ArrayBuffer]",ae="[object DataView]",ue="[object Float32Array]",ce="[object Float64Array]",se="[object Int8Array]",le="[object Int16Array]",fe="[object Int32Array]",pe="[object Uint8Array]",he="[object Uint8ClampedArray]",de="[object Uint16Array]",ve="[object Uint32Array]",ge=/\b__p \+= '';/g,me=/\b(__p \+=) '' \+/g,ye=/(__e\(.*?\)|\b__t\)) \+\n'';/g,_e=/&(?:amp|lt|gt|quot|#39);/g,be=/[&<>"']/g,xe=RegExp(_e.source),we=RegExp(be.source),Ce=/<%-([\s\S]+?)%>/g,ke=/<%([\s\S]+?)%>/g,Ee=/<%=([\s\S]+?)%>/g,Me=/\.|\[(?:[^[\]]*|(["'])(?:(?!\1)[^\\]|\\.)*?\1)\]/,Te=/^\w*$/,Se=/[^.[\]]+|\[(?:(-?\d+(?:\.\d+)?)|(["'])((?:(?!\2)[^\\]|\\.)*?)\2)\]|(?=(?:\.|\[\])(?:\.|\[\]|$))/g,Ne=/[\\^$.*+?()[\]{}|]/g,Ae=RegExp(Ne.source),Pe=/^\s+|\s+$/g,Oe=/^\s+/,Ie=/\s+$/,De=/\{(?:\n\/\* \[wrapped with .+\] \*\/)?\n?/,Re=/\{\n\/\* \[wrapped with (.+)\] \*/,Le=/,? & /,Ue=/[^\x00-\x2f\x3a-\x40\x5b-\x60\x7b-\x7f]+/g,Fe=/\\(\\)?/g,je=/\$\{([^\\}]*(?:\\.[^\\}]*)*)\}/g,Be=/\w*$/,Ve=/^[-+]0x[0-9a-f]+$/i,We=/^0b[01]+$/i,ze=/^\[object .+?Constructor\]$/,He=/^0o[0-7]+$/i,qe=/^(?:0|[1-9]\d*)$/,Ye=/[\xc0-\xd6\xd8-\xf6\xf8-\xff\u0100-\u017f]/g,Ke=/($^)/,Ge=/['\n\r\u2028\u2029\\]/g,$e="\\u0300-\\u036f\\ufe20-\\ufe2f\\u20d0-\\u20ff",Xe="\\xac\\xb1\\xd7\\xf7\\x00-\\x2f\\x3a-\\x40\\x5b-\\x60\\x7b-\\xbf\\u2000-\\u206f \\t\\x0b\\f\\xa0\\ufeff\\n\\r\\u2028\\u2029\\u1680\\u180e\\u2000\\u2001\\u2002\\u2003\\u2004\\u2005\\u2006\\u2007\\u2008\\u2009\\u200a\\u202f\\u205f\\u3000",Qe="["+Xe+"]",Ze="["+$e+"]",Je="[a-z\\xdf-\\xf6\\xf8-\\xff]",tn="[^\\ud800-\\udfff"+Xe+"\\d+\\u2700-\\u27bfa-z\\xdf-\\xf6\\xf8-\\xffA-Z\\xc0-\\xd6\\xd8-\\xde]",en="\\ud83c[\\udffb-\\udfff]",nn="(?:\\ud83c[\\udde6-\\uddff]){2}",rn="[\\ud800-\\udbff][\\udc00-\\udfff]",on="[A-Z\\xc0-\\xd6\\xd8-\\xde]",an="(?:"+Je+"|"+tn+")",un="(?:[\\u0300-\\u036f\\ufe20-\\ufe2f\\u20d0-\\u20ff]|\\ud83c[\\udffb-\\udfff])?",cn="(?:\\u200d(?:"+["[^\\ud800-\\udfff]",nn,rn].join("|")+")[\\ufe0e\\ufe0f]?"+un+")*",sn="[\\ufe0e\\ufe0f]?"+un+cn,ln="(?:"+["[\\u2700-\\u27bf]",nn,rn].join("|")+")"+sn,fn="(?:"+["[^\\ud800-\\udfff]"+Ze+"?",Ze,nn,rn,"[\\ud800-\\udfff]"].join("|")+")",pn=RegExp("['’]","g"),hn=RegExp(Ze,"g"),dn=RegExp(en+"(?="+en+")|"+fn+sn,"g"),vn=RegExp([on+"?"+Je+"+(?:['’](?:d|ll|m|re|s|t|ve))?(?="+[Qe,on,"$"].join("|")+")","(?:[A-Z\\xc0-\\xd6\\xd8-\\xde]|[^\\ud800-\\udfff\\xac\\xb1\\xd7\\xf7\\x00-\\x2f\\x3a-\\x40\\x5b-\\x60\\x7b-\\xbf\\u2000-\\u206f \\t\\x0b\\f\\xa0\\ufeff\\n\\r\\u2028\\u2029\\u1680\\u180e\\u2000\\u2001\\u2002\\u2003\\u2004\\u2005\\u2006\\u2007\\u2008\\u2009\\u200a\\u202f\\u205f\\u3000\\d+\\u2700-\\u27bfa-z\\xdf-\\xf6\\xf8-\\xffA-Z\\xc0-\\xd6\\xd8-\\xde])+(?:['’](?:D|LL|M|RE|S|T|VE))?(?="+[Qe,on+an,"$"].join("|")+")",on+"?"+an+"+(?:['’](?:d|ll|m|re|s|t|ve))?",on+"+(?:['’](?:D|LL|M|RE|S|T|VE))?","\\d*(?:1ST|2ND|3RD|(?![123])\\dTH)(?=\\b|[a-z_])","\\d*(?:1st|2nd|3rd|(?![123])\\dth)(?=\\b|[A-Z_])","\\d+",ln].join("|"),"g"),gn=RegExp("[\\u200d\\ud800-\\udfff"+$e+"\\ufe0e\\ufe0f]"),mn=/[a-z][A-Z]|[A-Z]{2}[a-z]|[0-9][a-zA-Z]|[a-zA-Z][0-9]|[^a-zA-Z0-9 ]/,yn=["Array","Buffer","DataView","Date","Error","Float32Array","Float64Array","Function","Int8Array","Int16Array","Int32Array","Map","Math","Object","Promise","RegExp","Set","String","Symbol","TypeError","Uint8Array","Uint8ClampedArray","Uint16Array","Uint32Array","WeakMap","_","clearTimeout","isFinite","parseInt","setTimeout"],_n=-1,bn={};bn[ue]=bn[ce]=bn[se]=bn[le]=bn[fe]=bn[pe]=bn[he]=bn[de]=bn[ve]=!0,bn[Ft]=bn[jt]=bn[oe]=bn[Vt]=bn[ae]=bn[Wt]=bn[Ht]=bn[qt]=bn[Kt]=bn[Gt]=bn[Xt]=bn[Zt]=bn[Jt]=bn[te]=bn[re]=!1;var xn={};xn[Ft]=xn[jt]=xn[oe]=xn[ae]=xn[Vt]=xn[Wt]=xn[ue]=xn[ce]=xn[se]=xn[le]=xn[fe]=xn[Kt]=xn[Gt]=xn[Xt]=xn[Zt]=xn[Jt]=xn[te]=xn[ee]=xn[pe]=xn[he]=xn[de]=xn[ve]=!0,xn[Ht]=xn[qt]=xn[re]=!1;var wn={"À":"A","Á":"A","Â":"A","Ã":"A","Ä":"A","Å":"A","à":"a","á":"a","â":"a","ã":"a","ä":"a","å":"a","Ç":"C","ç":"c","Ð":"D","ð":"d","È":"E","É":"E","Ê":"E","Ë":"E","è":"e","é":"e","ê":"e","ë":"e","Ì":"I","Í":"I","Î":"I","Ï":"I","ì":"i","í":"i","î":"i","ï":"i","Ñ":"N","ñ":"n","Ò":"O","Ó":"O","Ô":"O","Õ":"O","Ö":"O","Ø":"O","ò":"o","ó":"o","ô":"o","õ":"o","ö":"o","ø":"o","Ù":"U","Ú":"U","Û":"U","Ü":"U","ù":"u","ú":"u","û":"u","ü":"u","Ý":"Y","ý":"y","ÿ":"y","Æ":"Ae","æ":"ae","Þ":"Th","þ":"th","ß":"ss","Ā":"A","Ă":"A","Ą":"A","ā":"a","ă":"a","ą":"a","Ć":"C","Ĉ":"C","Ċ":"C","Č":"C","ć":"c","ĉ":"c","ċ":"c","č":"c","Ď":"D","Đ":"D","ď":"d","đ":"d","Ē":"E","Ĕ":"E","Ė":"E","Ę":"E","Ě":"E","ē":"e","ĕ":"e","ė":"e","ę":"e","ě":"e","Ĝ":"G","Ğ":"G","Ġ":"G","Ģ":"G","ĝ":"g","ğ":"g","ġ":"g","ģ":"g","Ĥ":"H","Ħ":"H","ĥ":"h","ħ":"h","Ĩ":"I","Ī":"I","Ĭ":"I","Į":"I","İ":"I","ĩ":"i","ī":"i","ĭ":"i","į":"i","ı":"i","Ĵ":"J","ĵ":"j","Ķ":"K","ķ":"k","ĸ":"k","Ĺ":"L","Ļ":"L","Ľ":"L","Ŀ":"L","Ł":"L","ĺ":"l","ļ":"l","ľ":"l","ŀ":"l","ł":"l","Ń":"N","Ņ":"N","Ň":"N","Ŋ":"N","ń":"n","ņ":"n","ň":"n","ŋ":"n","Ō":"O","Ŏ":"O","Ő":"O","ō":"o","ŏ":"o","ő":"o","Ŕ":"R","Ŗ":"R","Ř":"R","ŕ":"r","ŗ":"r","ř":"r","Ś":"S","Ŝ":"S","Ş":"S","Š":"S","ś":"s","ŝ":"s","ş":"s","š":"s","Ţ":"T","Ť":"T","Ŧ":"T","ţ":"t","ť":"t","ŧ":"t","Ũ":"U","Ū":"U","Ŭ":"U","Ů":"U","Ű":"U","Ų":"U","ũ":"u","ū":"u","ŭ":"u","ů":"u","ű":"u","ų":"u","Ŵ":"W","ŵ":"w","Ŷ":"Y","ŷ":"y","Ÿ":"Y","Ź":"Z","Ż":"Z","Ž":"Z","ź":"z","ż":"z","ž":"z","Ĳ":"IJ","ĳ":"ij","Œ":"Oe","œ":"oe","ŉ":"'n","ſ":"s"},Cn={"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"},kn={"&amp;":"&","&lt;":"<","&gt;":">","&quot;":'"',"&#39;":"'"},En={"\\":"\\","'":"'","\n":"n","\r":"r","\u2028":"u2028","\u2029":"u2029"},Mn=parseFloat,Tn=parseInt,Sn="object"==typeof t&&t&&t.Object===Object&&t,Nn="object"==typeof self&&self&&self.Object===Object&&self,An=Sn||Nn||Function("return this")(),Pn="object"==typeof e&&e&&!e.nodeType&&e,On=Pn&&"object"==typeof r&&r&&!r.nodeType&&r,In=On&&On.exports===Pn,Dn=In&&Sn.process,Rn=function(){try{var t=On&&On.require&&On.require("util").types;return t||Dn&&Dn.binding&&Dn.binding("util")}catch(t){}}(),Ln=Rn&&Rn.isArrayBuffer,Un=Rn&&Rn.isDate,Fn=Rn&&Rn.isMap,jn=Rn&&Rn.isRegExp,Bn=Rn&&Rn.isSet,Vn=Rn&&Rn.isTypedArray,Wn=M("length"),zn=T(wn),Hn=T(Cn),qn=T(kn),Yn=function t(e){function n(t){if(ec(t)&&!hp(t)&&!(t instanceof y)){if(t instanceof i)return t;if(pl.call(t,"__wrapped__"))return Zo(t)}return new i(t)}function r(){}function i(t,e){this.__wrapped__=t,this.__actions__=[],this.__chain__=!!e,this.__index__=0,this.__values__=nt}function y(t){this.__wrapped__=t,this.__actions__=[],this.__dir__=1,this.__filtered__=!1,this.__iteratees__=[],this.__takeCount__=Dt,this.__views__=[]}function T(){var t=new y(this.__wrapped__);return t.__actions__=Oi(this.__actions__),t.__dir__=this.__dir__,t.__filtered__=this.__filtered__,t.__iteratees__=Oi(this.__iteratees__),t.__takeCount__=this.__takeCount__,t.__views__=Oi(this.__views__),t}function $(){if(this.__filtered__){var t=new y(this);t.__dir__=-1,t.__filtered__=!0}else t=this.clone(),t.__dir__*=-1;return t}function J(){var t=this.__wrapped__.value(),e=this.__dir__,n=hp(t),r=e<0,i=n?t.length:0,o=wo(0,i,this.__views__),a=o.start,u=o.end,c=u-a,s=r?u:a-1,l=this.__iteratees__,f=l.length,p=0,h=Wl(c,this.__takeCount__);if(!n||!r&&i==c&&h==c)return vi(t,this.__actions__);var d=[];t:for(;c--&&p<h;){s+=e;for(var v=-1,g=t[s];++v<f;){var m=l[v],y=m.iteratee,_=m.type,b=y(g);if(_==Nt)g=b;else if(!b){if(_==St)continue t;break t}}d[p++]=g}return d}function tt(t){var e=-1,n=null==t?0:t.length;for(this.clear();++e<n;){var r=t[e];this.set(r[0],r[1])}}function Ue(){this.__data__=Zl?Zl(null):{},this.size=0}function $e(t){var e=this.has(t)&&delete this.__data__[t];return this.size-=e?1:0,e}function Xe(t){var e=this.__data__;if(Zl){var n=e[t];return n===at?nt:n}return pl.call(e,t)?e[t]:nt}function Qe(t){var e=this.__data__;return Zl?e[t]!==nt:pl.call(e,t)}function Ze(t,e){var n=this.__data__;return this.size+=this.has(t)?0:1,n[t]=Zl&&e===nt?at:e,this}function Je(t){var e=-1,n=null==t?0:t.length;for(this.clear();++e<n;){var r=t[e];this.set(r[0],r[1])}}function tn(){this.__data__=[],this.size=0}function en(t){var e=this.__data__,n=Kn(e,t);return!(n<0)&&(n==e.length-1?e.pop():Ml.call(e,n,1),--this.size,!0)}function nn(t){var e=this.__data__,n=Kn(e,t);return n<0?nt:e[n][1]}function rn(t){return Kn(this.__data__,t)>-1}function on(t,e){var n=this.__data__,r=Kn(n,t);return r<0?(++this.size,n.push([t,e])):n[r][1]=e,this}function an(t){var e=-1,n=null==t?0:t.length;for(this.clear();++e<n;){var r=t[e];this.set(r[0],r[1])}}function un(){this.size=0,this.__data__={hash:new tt,map:new(Gl||Je),string:new tt}}function cn(t){var e=yo(this,t).delete(t);return this.size-=e?1:0,e}function sn(t){return yo(this,t).get(t)}function ln(t){return yo(this,t).has(t)}function fn(t,e){var n=yo(this,t),r=n.size;return n.set(t,e),this.size+=n.size==r?0:1,this}function dn(t){var e=-1,n=null==t?0:t.length;for(this.__data__=new an;++e<n;)this.add(t[e])}function vn(t){return this.__data__.set(t,at),this}function gn(t){return this.__data__.has(t)}function mn(t){var e=this.__data__=new Je(t);this.size=e.size}function wn(){this.__data__=new Je,this.size=0}function Cn(t){var e=this.__data__,n=e.delete(t);return this.size=e.size,n}function kn(t){return this.__data__.get(t)}function En(t){return this.__data__.has(t)}function Sn(t,e){var n=this.__data__;if(n instanceof Je){var r=n.__data__;if(!Gl||r.length<rt-1)return r.push([t,e]),this.size=++n.size,this;n=this.__data__=new an(r)}return n.set(t,e),this.size=n.size,this}function Nn(t,e){var n=hp(t),r=!n&&pp(t),i=!n&&!r&&vp(t),o=!n&&!r&&!i&&bp(t),a=n||r||i||o,u=a?P(t.length,ol):[],c=u.length;for(var s in t)!e&&!pl.call(t,s)||a&&("length"==s||i&&("offset"==s||"parent"==s)||o&&("buffer"==s||"byteLength"==s||"byteOffset"==s)||Ao(s,c))||u.push(s);return u}function Pn(t){var e=t.length;return e?t[Xr(0,e-1)]:nt}function On(t,e){return Go(Oi(t),Jn(e,0,t.length))}function Dn(t){return Go(Oi(t))}function Rn(t,e,n){(n===nt||Vu(t[e],n))&&(n!==nt||e in t)||Qn(t,e,n)}function Wn(t,e,n){var r=t[e];pl.call(t,e)&&Vu(r,n)&&(n!==nt||e in t)||Qn(t,e,n)}function Kn(t,e){for(var n=t.length;n--;)if(Vu(t[n][0],e))return n;return-1}function Gn(t,e,n,r){return ff(t,function(t,i,o){e(r,t,n(t),o)}),r}function $n(t,e){return t&&Ii(e,Lc(e),t)}function Xn(t,e){return t&&Ii(e,Uc(e),t)}function Qn(t,e,n){"__proto__"==e&&Al?Al(t,e,{configurable:!0,enumerable:!0,value:n,writable:!0}):t[e]=n}function Zn(t,e){for(var n=-1,r=e.length,i=Zs(r),o=null==t;++n<r;)i[n]=o?nt:Ic(t,e[n]);return i}function Jn(t,e,n){return t===t&&(n!==nt&&(t=t<=n?t:n),e!==nt&&(t=t>=e?t:e)),t}function tr(t,e,n,r,i,o){var a,c=e&st,s=e&lt,l=e&ft;if(n&&(a=i?n(t,r,i,o):n(t)),a!==nt)return a;if(!tc(t))return t;var f=hp(t);if(f){if(a=Eo(t),!c)return Oi(t,a)}else{var p=Cf(t),h=p==qt||p==Yt;if(vp(t))return wi(t,c);if(p==Xt||p==Ft||h&&!i){if(a=s||h?{}:Mo(t),!c)return s?Ri(t,Xn(a,t)):Di(t,$n(a,t))}else{if(!xn[p])return i?t:{};a=To(t,p,c)}}o||(o=new mn);var d=o.get(t);if(d)return d;if(o.set(t,a),_p(t))return t.forEach(function(r){a.add(tr(r,e,n,r,t,o))}),a;if(mp(t))return t.forEach(function(r,i){a.set(i,tr(r,e,n,i,t,o))}),a;var v=l?s?ho:po:s?Uc:Lc,g=f?nt:v(t);return u(g||t,function(r,i){g&&(i=r,r=t[i]),Wn(a,i,tr(r,e,n,i,t,o))}),a}function er(t){var e=Lc(t);return function(n){return nr(n,t,e)}}function nr(t,e,n){var r=n.length;if(null==t)return!r;for(t=rl(t);r--;){var i=n[r],o=e[i],a=t[i];if(a===nt&&!(i in t)||!o(a))return!1}return!0}function rr(t,e,n){if("function"!=typeof t)throw new al(ot);return Mf(function(){t.apply(nt,n)},e)}function ir(t,e,n,r){var i=-1,o=f,a=!0,u=t.length,c=[],s=e.length;if(!u)return c;n&&(e=h(e,I(n))),r?(o=p,a=!1):e.length>=rt&&(o=R,a=!1,e=new dn(e));t:for(;++i<u;){var l=t[i],d=null==n?l:n(l);if(l=r||0!==l?l:0,a&&d===d){for(var v=s;v--;)if(e[v]===d)continue t;c.push(l)}else o(e,d,r)||c.push(l)}return c}function or(t,e){var n=!0;return ff(t,function(t,r,i){return n=!!e(t,r,i)}),n}function ar(t,e,n){for(var r=-1,i=t.length;++r<i;){var o=t[r],a=e(o);if(null!=a&&(u===nt?a===a&&!pc(a):n(a,u)))var u=a,c=o}return c}function ur(t,e,n,r){var i=t.length;for(n=yc(n),n<0&&(n=-n>i?0:i+n),r=r===nt||r>i?i:yc(r),r<0&&(r+=i),r=n>r?0:_c(r);n<r;)t[n++]=e;return t}function cr(t,e){var n=[];return ff(t,function(t,r,i){e(t,r,i)&&n.push(t)}),n}function sr(t,e,n,r,i){var o=-1,a=t.length;for(n||(n=No),i||(i=[]);++o<a;){var u=t[o];e>0&&n(u)?e>1?sr(u,e-1,n,r,i):d(i,u):r||(i[i.length]=u)}return i}function lr(t,e){return t&&hf(t,e,Lc)}function fr(t,e){return t&&df(t,e,Lc)}function pr(t,e){return l(e,function(e){return Qu(t[e])})}function hr(t,e){e=bi(e,t);for(var n=0,r=e.length;null!=t&&n<r;)t=t[$o(e[n++])];return n&&n==r?t:nt}function dr(t,e,n){var r=e(t);return hp(t)?r:d(r,n(t))}function vr(t){return null==t?t===nt?ne:$t:Nl&&Nl in rl(t)?xo(t):Vo(t)}function gr(t,e){return t>e}function mr(t,e){return null!=t&&pl.call(t,e)}function yr(t,e){return null!=t&&e in rl(t)}function _r(t,e,n){return t>=Wl(e,n)&&t<Vl(e,n)}function br(t,e,n){for(var r=n?p:f,i=t[0].length,o=t.length,a=o,u=Zs(o),c=1/0,s=[];a--;){var l=t[a];a&&e&&(l=h(l,I(e))),c=Wl(l.length,c),u[a]=!n&&(e||i>=120&&l.length>=120)?new dn(a&&l):nt}l=t[0];var d=-1,v=u[0];t:for(;++d<i&&s.length<c;){var g=l[d],m=e?e(g):g;if(g=n||0!==g?g:0,!(v?R(v,m):r(s,m,n))){for(a=o;--a;){var y=u[a];if(!(y?R(y,m):r(t[a],m,n)))continue t}v&&v.push(m),s.push(g)}}return s}function xr(t,e,n,r){return lr(t,function(t,i,o){e(r,n(t),i,o)}),r}function wr(t,e,n){e=bi(e,t),t=zo(t,e);var r=null==t?t:t[$o(ma(e))];return null==r?nt:o(r,t,n)}function Cr(t){return ec(t)&&vr(t)==Ft}function kr(t){return ec(t)&&vr(t)==oe}function Er(t){return ec(t)&&vr(t)==Wt}function Mr(t,e,n,r,i){return t===e||(null==t||null==e||!ec(t)&&!ec(e)?t!==t&&e!==e:Tr(t,e,n,r,Mr,i))}function Tr(t,e,n,r,i,o){var a=hp(t),u=hp(e),c=a?jt:Cf(t),s=u?jt:Cf(e);c=c==Ft?Xt:c,s=s==Ft?Xt:s;var l=c==Xt,f=s==Xt,p=c==s;if(p&&vp(t)){if(!vp(e))return!1;a=!0,l=!1}if(p&&!l)return o||(o=new mn),a||bp(t)?co(t,e,n,r,i,o):so(t,e,c,n,r,i,o);if(!(n&pt)){var h=l&&pl.call(t,"__wrapped__"),d=f&&pl.call(e,"__wrapped__");if(h||d){var v=h?t.value():t,g=d?e.value():e;return o||(o=new mn),i(v,g,n,r,o)}}return!!p&&(o||(o=new mn),lo(t,e,n,r,i,o))}function Sr(t){return ec(t)&&Cf(t)==Kt}function Nr(t,e,n,r){var i=n.length,o=i,a=!r;if(null==t)return!o;for(t=rl(t);i--;){var u=n[i];if(a&&u[2]?u[1]!==t[u[0]]:!(u[0]in t))return!1}for(;++i<o;){u=n[i];var c=u[0],s=t[c],l=u[1];if(a&&u[2]){if(s===nt&&!(c in t))return!1}else{var f=new mn;if(r)var p=r(s,l,c,t,e,f);if(!(p===nt?Mr(l,s,pt|ht,r,f):p))return!1}}return!0}function Ar(t){return!(!tc(t)||Ro(t))&&(Qu(t)?yl:ze).test(Xo(t))}function Pr(t){return ec(t)&&vr(t)==Zt}function Or(t){return ec(t)&&Cf(t)==Jt}function Ir(t){return ec(t)&&Ju(t.length)&&!!bn[vr(t)]}function Dr(t){return"function"==typeof t?t:null==t?Ms:"object"==typeof t?hp(t)?Br(t[0],t[1]):jr(t):Ds(t)}function Rr(t){if(!Lo(t))return Bl(t);var e=[];for(var n in rl(t))pl.call(t,n)&&"constructor"!=n&&e.push(n);return e}function Lr(t){if(!tc(t))return Bo(t);var e=Lo(t),n=[];for(var r in t)("constructor"!=r||!e&&pl.call(t,r))&&n.push(r);return n}function Ur(t,e){return t<e}function Fr(t,e){var n=-1,r=Wu(t)?Zs(t.length):[];return ff(t,function(t,i,o){r[++n]=e(t,i,o)}),r}function jr(t){var e=_o(t);return 1==e.length&&e[0][2]?Fo(e[0][0],e[0][1]):function(n){return n===t||Nr(n,t,e)}}function Br(t,e){return Oo(t)&&Uo(e)?Fo($o(t),e):function(n){var r=Ic(n,t);return r===nt&&r===e?Rc(n,t):Mr(e,r,pt|ht)}}function Vr(t,e,n,r,i){t!==e&&hf(e,function(o,a){if(tc(o))i||(i=new mn),Wr(t,e,a,n,Vr,r,i);else{var u=r?r(qo(t,a),o,a+"",t,e,i):nt;u===nt&&(u=o),Rn(t,a,u)}},Uc)}function Wr(t,e,n,r,i,o,a){var u=qo(t,n),c=qo(e,n),s=a.get(c);if(s)return void Rn(t,n,s);var l=o?o(u,c,n+"",t,e,a):nt,f=l===nt;if(f){var p=hp(c),h=!p&&vp(c),d=!p&&!h&&bp(c);l=c,p||h||d?hp(u)?l=u:zu(u)?l=Oi(u):h?(f=!1,l=wi(c,!0)):d?(f=!1,l=Ti(c,!0)):l=[]:sc(c)||pp(c)?(l=u,pp(u)?l=xc(u):tc(u)&&!Qu(u)||(l=Mo(c))):f=!1}f&&(a.set(c,l),i(l,c,r,o,a),a.delete(c)),Rn(t,n,l)}function zr(t,e){var n=t.length;if(n)return e+=e<0?n:0,Ao(e,n)?t[e]:nt}function Hr(t,e,n){var r=-1;return e=h(e.length?e:[Ms],I(mo())),N(Fr(t,function(t,n,i){return{criteria:h(e,function(e){return e(t)}),index:++r,value:t}}),function(t,e){return Ni(t,e,n)})}function qr(t,e){return Yr(t,e,function(e,n){return Rc(t,n)})}function Yr(t,e,n){for(var r=-1,i=e.length,o={};++r<i;){var a=e[r],u=hr(t,a);n(u,a)&&ni(o,bi(a,t),u)}return o}function Kr(t){return function(e){return hr(e,t)}}function Gr(t,e,n,r){var i=r?C:w,o=-1,a=e.length,u=t;for(t===e&&(e=Oi(e)),n&&(u=h(t,I(n)));++o<a;)for(var c=0,s=e[o],l=n?n(s):s;(c=i(u,l,c,r))>-1;)u!==t&&Ml.call(u,c,1),Ml.call(t,c,1);return t}function $r(t,e){for(var n=t?e.length:0,r=n-1;n--;){var i=e[n];if(n==r||i!==o){var o=i;Ao(i)?Ml.call(t,i,1):pi(t,i)}}return t}function Xr(t,e){return t+Rl(ql()*(e-t+1))}function Qr(t,e,n,r){for(var i=-1,o=Vl(Dl((e-t)/(n||1)),0),a=Zs(o);o--;)a[r?o:++i]=t,t+=n;return a}function Zr(t,e){var n="";if(!t||e<1||e>Pt)return n;do{e%2&&(n+=t),(e=Rl(e/2))&&(t+=t)}while(e);return n}function Jr(t,e){return Tf(Wo(t,e,Ms),t+"")}function ti(t){return Pn($c(t))}function ei(t,e){var n=$c(t);return Go(n,Jn(e,0,n.length))}function ni(t,e,n,r){if(!tc(t))return t;e=bi(e,t);for(var i=-1,o=e.length,a=o-1,u=t;null!=u&&++i<o;){var c=$o(e[i]),s=n;if(i!=a){var l=u[c];s=r?r(l,c,u):nt,s===nt&&(s=tc(l)?l:Ao(e[i+1])?[]:{})}Wn(u,c,s),u=u[c]}return t}function ri(t){return Go($c(t))}function ii(t,e,n){var r=-1,i=t.length;e<0&&(e=-e>i?0:i+e),n=n>i?i:n,n<0&&(n+=i),i=e>n?0:n-e>>>0,e>>>=0;for(var o=Zs(i);++r<i;)o[r]=t[r+e];return o}function oi(t,e){var n;return ff(t,function(t,r,i){return!(n=e(t,r,i))}),!!n}function ai(t,e,n){var r=0,i=null==t?r:t.length;if("number"==typeof e&&e===e&&i<=Lt){for(;r<i;){var o=r+i>>>1,a=t[o];null!==a&&!pc(a)&&(n?a<=e:a<e)?r=o+1:i=o}return i}return ui(t,e,Ms,n)}function ui(t,e,n,r){e=n(e);for(var i=0,o=null==t?0:t.length,a=e!==e,u=null===e,c=pc(e),s=e===nt;i<o;){var l=Rl((i+o)/2),f=n(t[l]),p=f!==nt,h=null===f,d=f===f,v=pc(f);if(a)var g=r||d;else g=s?d&&(r||p):u?d&&p&&(r||!h):c?d&&p&&!h&&(r||!v):!h&&!v&&(r?f<=e:f<e);g?i=l+1:o=l}return Wl(o,Rt)}function ci(t,e){for(var n=-1,r=t.length,i=0,o=[];++n<r;){var a=t[n],u=e?e(a):a;if(!n||!Vu(u,c)){var c=u;o[i++]=0===a?0:a}}return o}function si(t){return"number"==typeof t?t:pc(t)?It:+t}function li(t){if("string"==typeof t)return t;if(hp(t))return h(t,li)+"";if(pc(t))return sf?sf.call(t):"";var e=t+"";return"0"==e&&1/t==-At?"-0":e}function fi(t,e,n){var r=-1,i=f,o=t.length,a=!0,u=[],c=u;if(n)a=!1,i=p;else if(o>=rt){var s=e?null:_f(t);if(s)return K(s);a=!1,i=R,c=new dn}else c=e?[]:u;t:for(;++r<o;){var l=t[r],h=e?e(l):l;if(l=n||0!==l?l:0,a&&h===h){for(var d=c.length;d--;)if(c[d]===h)continue t;e&&c.push(h),u.push(l)}else i(c,h,n)||(c!==u&&c.push(h),u.push(l))}return u}function pi(t,e){return e=bi(e,t),null==(t=zo(t,e))||delete t[$o(ma(e))]}function hi(t,e,n,r){return ni(t,e,n(hr(t,e)),r)}function di(t,e,n,r){for(var i=t.length,o=r?i:-1;(r?o--:++o<i)&&e(t[o],o,t););return n?ii(t,r?0:o,r?o+1:i):ii(t,r?o+1:0,r?i:o)}function vi(t,e){var n=t;return n instanceof y&&(n=n.value()),v(e,function(t,e){return e.func.apply(e.thisArg,d([t],e.args))},n)}function gi(t,e,n){var r=t.length;if(r<2)return r?fi(t[0]):[];for(var i=-1,o=Zs(r);++i<r;)for(var a=t[i],u=-1;++u<r;)u!=i&&(o[i]=ir(o[i]||a,t[u],e,n));return fi(sr(o,1),e,n)}function mi(t,e,n){for(var r=-1,i=t.length,o=e.length,a={};++r<i;){var u=r<o?e[r]:nt;n(a,t[r],u)}return a}function yi(t){return zu(t)?t:[]}function _i(t){return"function"==typeof t?t:Ms}function bi(t,e){return hp(t)?t:Oo(t,e)?[t]:Sf(Cc(t))}function xi(t,e,n){var r=t.length;return n=n===nt?r:n,!e&&n>=r?t:ii(t,e,n)}function wi(t,e){if(e)return t.slice();var n=t.length,r=wl?wl(n):new t.constructor(n);return t.copy(r),r}function Ci(t){var e=new t.constructor(t.byteLength);return new xl(e).set(new xl(t)),e}function ki(t,e){var n=e?Ci(t.buffer):t.buffer;return new t.constructor(n,t.byteOffset,t.byteLength)}function Ei(t){var e=new t.constructor(t.source,Be.exec(t));return e.lastIndex=t.lastIndex,e}function Mi(t){return cf?rl(cf.call(t)):{}}function Ti(t,e){var n=e?Ci(t.buffer):t.buffer;return new t.constructor(n,t.byteOffset,t.length)}function Si(t,e){if(t!==e){var n=t!==nt,r=null===t,i=t===t,o=pc(t),a=e!==nt,u=null===e,c=e===e,s=pc(e);if(!u&&!s&&!o&&t>e||o&&a&&c&&!u&&!s||r&&a&&c||!n&&c||!i)return 1;if(!r&&!o&&!s&&t<e||s&&n&&i&&!r&&!o||u&&n&&i||!a&&i||!c)return-1}return 0}function Ni(t,e,n){for(var r=-1,i=t.criteria,o=e.criteria,a=i.length,u=n.length;++r<a;){var c=Si(i[r],o[r]);if(c){if(r>=u)return c;return c*("desc"==n[r]?-1:1)}}return t.index-e.index}function Ai(t,e,n,r){for(var i=-1,o=t.length,a=n.length,u=-1,c=e.length,s=Vl(o-a,0),l=Zs(c+s),f=!r;++u<c;)l[u]=e[u];for(;++i<a;)(f||i<o)&&(l[n[i]]=t[i]);for(;s--;)l[u++]=t[i++];return l}function Pi(t,e,n,r){for(var i=-1,o=t.length,a=-1,u=n.length,c=-1,s=e.length,l=Vl(o-u,0),f=Zs(l+s),p=!r;++i<l;)f[i]=t[i];for(var h=i;++c<s;)f[h+c]=e[c];for(;++a<u;)(p||i<o)&&(f[h+n[a]]=t[i++]);return f}function Oi(t,e){var n=-1,r=t.length;for(e||(e=Zs(r));++n<r;)e[n]=t[n];return e}function Ii(t,e,n,r){var i=!n;n||(n={});for(var o=-1,a=e.length;++o<a;){var u=e[o],c=r?r(n[u],t[u],u,n,t):nt;c===nt&&(c=t[u]),i?Qn(n,u,c):Wn(n,u,c)}return n}function Di(t,e){return Ii(t,xf(t),e)}function Ri(t,e){return Ii(t,wf(t),e)}function Li(t,e){return function(n,r){var i=hp(n)?a:Gn,o=e?e():{};return i(n,t,mo(r,2),o)}}function Ui(t){return Jr(function(e,n){var r=-1,i=n.length,o=i>1?n[i-1]:nt,a=i>2?n[2]:nt;for(o=t.length>3&&"function"==typeof o?(i--,o):nt,a&&Po(n[0],n[1],a)&&(o=i<3?nt:o,i=1),e=rl(e);++r<i;){var u=n[r];u&&t(e,u,r,o)}return e})}function Fi(t,e){return function(n,r){if(null==n)return n;if(!Wu(n))return t(n,r);for(var i=n.length,o=e?i:-1,a=rl(n);(e?o--:++o<i)&&!1!==r(a[o],o,a););return n}}function ji(t){return function(e,n,r){for(var i=-1,o=rl(e),a=r(e),u=a.length;u--;){var c=a[t?u:++i];if(!1===n(o[c],c,o))break}return e}}function Bi(t,e,n){function r(){return(this&&this!==An&&this instanceof r?o:t).apply(i?n:this,arguments)}var i=e&dt,o=zi(t);return r}function Vi(t){return function(e){e=Cc(e);var n=V(e)?Z(e):nt,r=n?n[0]:e.charAt(0),i=n?xi(n,1).join(""):e.slice(1);return r[t]()+i}}function Wi(t){return function(e){return v(xs(es(e).replace(pn,"")),t,"")}}function zi(t){return function(){var e=arguments;switch(e.length){case 0:return new t;case 1:return new t(e[0]);case 2:return new t(e[0],e[1]);case 3:return new t(e[0],e[1],e[2]);case 4:return new t(e[0],e[1],e[2],e[3]);case 5:return new t(e[0],e[1],e[2],e[3],e[4]);case 6:return new t(e[0],e[1],e[2],e[3],e[4],e[5]);case 7:return new t(e[0],e[1],e[2],e[3],e[4],e[5],e[6])}var n=lf(t.prototype),r=t.apply(n,e);return tc(r)?r:n}}function Hi(t,e,n){function r(){for(var a=arguments.length,u=Zs(a),c=a,s=go(r);c--;)u[c]=arguments[c];var l=a<3&&u[0]!==s&&u[a-1]!==s?[]:Y(u,s);return(a-=l.length)<n?eo(t,e,Ki,r.placeholder,nt,u,l,nt,nt,n-a):o(this&&this!==An&&this instanceof r?i:t,this,u)}var i=zi(t);return r}function qi(t){return function(e,n,r){var i=rl(e);if(!Wu(e)){var o=mo(n,3);e=Lc(e),n=function(t){return o(i[t],t,i)}}var a=t(e,n,r);return a>-1?i[o?e[a]:a]:nt}}function Yi(t){return fo(function(e){var n=e.length,r=n,o=i.prototype.thru;for(t&&e.reverse();r--;){var a=e[r];if("function"!=typeof a)throw new al(ot);if(o&&!u&&"wrapper"==vo(a))var u=new i([],!0)}for(r=u?r:n;++r<n;){a=e[r];var c=vo(a),s="wrapper"==c?bf(a):nt;u=s&&Do(s[0])&&s[1]==(xt|mt|_t|wt)&&!s[4].length&&1==s[9]?u[vo(s[0])].apply(u,s[3]):1==a.length&&Do(a)?u[c]():u.thru(a)}return function(){var t=arguments,r=t[0];if(u&&1==t.length&&hp(r))return u.plant(r).value();for(var i=0,o=n?e[i].apply(this,t):r;++i<n;)o=e[i].call(this,o);return o}})}function Ki(t,e,n,r,i,o,a,u,c,s){function l(){for(var m=arguments.length,y=Zs(m),_=m;_--;)y[_]=arguments[_];if(d)var b=go(l),x=F(y,b);if(r&&(y=Ai(y,r,i,d)),o&&(y=Pi(y,o,a,d)),m-=x,d&&m<s){var w=Y(y,b);return eo(t,e,Ki,l.placeholder,n,y,w,u,c,s-m)}var C=p?n:this,k=h?C[t]:t;return m=y.length,u?y=Ho(y,u):v&&m>1&&y.reverse(),f&&c<m&&(y.length=c),this&&this!==An&&this instanceof l&&(k=g||zi(k)),k.apply(C,y)}var f=e&xt,p=e&dt,h=e&vt,d=e&(mt|yt),v=e&Ct,g=h?nt:zi(t);return l}function Gi(t,e){return function(n,r){return xr(n,t,e(r),{})}}function $i(t,e){return function(n,r){var i;if(n===nt&&r===nt)return e;if(n!==nt&&(i=n),r!==nt){if(i===nt)return r;"string"==typeof n||"string"==typeof r?(n=li(n),r=li(r)):(n=si(n),r=si(r)),i=t(n,r)}return i}}function Xi(t){return fo(function(e){return e=h(e,I(mo())),Jr(function(n){var r=this;return t(e,function(t){return o(t,r,n)})})})}function Qi(t,e){e=e===nt?" ":li(e);var n=e.length;if(n<2)return n?Zr(e,t):e;var r=Zr(e,Dl(t/Q(e)));return V(e)?xi(Z(r),0,t).join(""):r.slice(0,t)}function Zi(t,e,n,r){function i(){for(var e=-1,c=arguments.length,s=-1,l=r.length,f=Zs(l+c),p=this&&this!==An&&this instanceof i?u:t;++s<l;)f[s]=r[s];for(;c--;)f[s++]=arguments[++e];return o(p,a?n:this,f)}var a=e&dt,u=zi(t);return i}function Ji(t){return function(e,n,r){return r&&"number"!=typeof r&&Po(e,n,r)&&(n=r=nt),e=mc(e),n===nt?(n=e,e=0):n=mc(n),r=r===nt?e<n?1:-1:mc(r),Qr(e,n,r,t)}}function to(t){return function(e,n){return"string"==typeof e&&"string"==typeof n||(e=bc(e),n=bc(n)),t(e,n)}}function eo(t,e,n,r,i,o,a,u,c,s){var l=e&mt,f=l?a:nt,p=l?nt:a,h=l?o:nt,d=l?nt:o;e|=l?_t:bt,(e&=~(l?bt:_t))&gt||(e&=~(dt|vt));var v=[t,e,i,h,f,d,p,u,c,s],g=n.apply(nt,v);return Do(t)&&Ef(g,v),g.placeholder=r,Yo(g,t,e)}function no(t){var e=nl[t];return function(t,n){if(t=bc(t),n=null==n?0:Wl(yc(n),292)){var r=(Cc(t)+"e").split("e");return r=(Cc(e(r[0]+"e"+(+r[1]+n)))+"e").split("e"),+(r[0]+"e"+(+r[1]-n))}return e(t)}}function ro(t){return function(e){var n=Cf(e);return n==Kt?H(e):n==Jt?G(e):O(e,t(e))}}function io(t,e,n,r,i,o,a,u){var c=e&vt;if(!c&&"function"!=typeof t)throw new al(ot);var s=r?r.length:0;if(s||(e&=~(_t|bt),r=i=nt),a=a===nt?a:Vl(yc(a),0),u=u===nt?u:yc(u),s-=i?i.length:0,e&bt){var l=r,f=i;r=i=nt}var p=c?nt:bf(t),h=[t,e,n,r,i,l,f,o,a,u];if(p&&jo(h,p),t=h[0],e=h[1],n=h[2],r=h[3],i=h[4],u=h[9]=h[9]===nt?c?0:t.length:Vl(h[9]-s,0),!u&&e&(mt|yt)&&(e&=~(mt|yt)),e&&e!=dt)d=e==mt||e==yt?Hi(t,e,u):e!=_t&&e!=(dt|_t)||i.length?Ki.apply(nt,h):Zi(t,e,n,r);else var d=Bi(t,e,n);return Yo((p?vf:Ef)(d,h),t,e)}function oo(t,e,n,r){return t===nt||Vu(t,sl[n])&&!pl.call(r,n)?e:t}function ao(t,e,n,r,i,o){return tc(t)&&tc(e)&&(o.set(e,t),Vr(t,e,nt,ao,o),o.delete(e)),t}function uo(t){return sc(t)?nt:t}function co(t,e,n,r,i,o){var a=n&pt,u=t.length,c=e.length;if(u!=c&&!(a&&c>u))return!1;var s=o.get(t);if(s&&o.get(e))return s==e;var l=-1,f=!0,p=n&ht?new dn:nt;for(o.set(t,e),o.set(e,t);++l<u;){var h=t[l],d=e[l];if(r)var v=a?r(d,h,l,e,t,o):r(h,d,l,t,e,o);if(v!==nt){if(v)continue;f=!1;break}if(p){if(!m(e,function(t,e){if(!R(p,e)&&(h===t||i(h,t,n,r,o)))return p.push(e)})){f=!1;break}}else if(h!==d&&!i(h,d,n,r,o)){f=!1;break}}return o.delete(t),o.delete(e),f}function so(t,e,n,r,i,o,a){switch(n){case ae:if(t.byteLength!=e.byteLength||t.byteOffset!=e.byteOffset)return!1;t=t.buffer,e=e.buffer;case oe:return!(t.byteLength!=e.byteLength||!o(new xl(t),new xl(e)));case Vt:case Wt:case Gt:return Vu(+t,+e);case Ht:return t.name==e.name&&t.message==e.message;case Zt:case te:return t==e+"";case Kt:var u=H;case Jt:var c=r&pt;if(u||(u=K),t.size!=e.size&&!c)return!1;var s=a.get(t);if(s)return s==e;r|=ht,a.set(t,e);var l=co(u(t),u(e),r,i,o,a);return a.delete(t),l;case ee:if(cf)return cf.call(t)==cf.call(e)}return!1}function lo(t,e,n,r,i,o){var a=n&pt,u=po(t),c=u.length;if(c!=po(e).length&&!a)return!1;for(var s=c;s--;){var l=u[s];if(!(a?l in e:pl.call(e,l)))return!1}var f=o.get(t);if(f&&o.get(e))return f==e;var p=!0;o.set(t,e),o.set(e,t);for(var h=a;++s<c;){l=u[s];var d=t[l],v=e[l];if(r)var g=a?r(v,d,l,e,t,o):r(d,v,l,t,e,o);if(!(g===nt?d===v||i(d,v,n,r,o):g)){p=!1;break}h||(h="constructor"==l)}if(p&&!h){var m=t.constructor,y=e.constructor;m!=y&&"constructor"in t&&"constructor"in e&&!("function"==typeof m&&m instanceof m&&"function"==typeof y&&y instanceof y)&&(p=!1)}return o.delete(t),o.delete(e),p}function fo(t){return Tf(Wo(t,nt,sa),t+"")}function po(t){return dr(t,Lc,xf)}function ho(t){return dr(t,Uc,wf)}function vo(t){for(var e=t.name+"",n=tf[e],r=pl.call(tf,e)?n.length:0;r--;){var i=n[r],o=i.func;if(null==o||o==t)return i.name}return e}function go(t){return(pl.call(n,"placeholder")?n:t).placeholder}function mo(){var t=n.iteratee||Ts;return t=t===Ts?Dr:t,arguments.length?t(arguments[0],arguments[1]):t}function yo(t,e){var n=t.__data__;return Io(e)?n["string"==typeof e?"string":"hash"]:n.map}function _o(t){for(var e=Lc(t),n=e.length;n--;){var r=e[n],i=t[r];e[n]=[r,i,Uo(i)]}return e}function bo(t,e){var n=B(t,e);return Ar(n)?n:nt}function xo(t){var e=pl.call(t,Nl),n=t[Nl];try{t[Nl]=nt;var r=!0}catch(t){}var i=vl.call(t);return r&&(e?t[Nl]=n:delete t[Nl]),i}function wo(t,e,n){for(var r=-1,i=n.length;++r<i;){var o=n[r],a=o.size;switch(o.type){case"drop":t+=a;break;case"dropRight":e-=a;break;case"take":e=Wl(e,t+a);break;case"takeRight":t=Vl(t,e-a)}}return{start:t,end:e}}function Co(t){var e=t.match(Re);return e?e[1].split(Le):[]}function ko(t,e,n){e=bi(e,t);for(var r=-1,i=e.length,o=!1;++r<i;){var a=$o(e[r]);if(!(o=null!=t&&n(t,a)))break;t=t[a]}return o||++r!=i?o:!!(i=null==t?0:t.length)&&Ju(i)&&Ao(a,i)&&(hp(t)||pp(t))}function Eo(t){var e=t.length,n=new t.constructor(e);return e&&"string"==typeof t[0]&&pl.call(t,"index")&&(n.index=t.index,n.input=t.input),n}function Mo(t){return"function"!=typeof t.constructor||Lo(t)?{}:lf(Cl(t))}function To(t,e,n){var r=t.constructor;switch(e){case oe:return Ci(t);case Vt:case Wt:return new r(+t);case ae:return ki(t,n);case ue:case ce:case se:case le:case fe:case pe:case he:case de:case ve:return Ti(t,n);case Kt:return new r;case Gt:case te:return new r(t);case Zt:return Ei(t);case Jt:return new r;case ee:return Mi(t)}}function So(t,e){var n=e.length;if(!n)return t;var r=n-1;return e[r]=(n>1?"& ":"")+e[r],e=e.join(n>2?", ":" "),t.replace(De,"{\n/* [wrapped with "+e+"] */\n")}function No(t){return hp(t)||pp(t)||!!(Tl&&t&&t[Tl])}function Ao(t,e){var n=typeof t;return!!(e=null==e?Pt:e)&&("number"==n||"symbol"!=n&&qe.test(t))&&t>-1&&t%1==0&&t<e}function Po(t,e,n){if(!tc(n))return!1;var r=typeof e;return!!("number"==r?Wu(n)&&Ao(e,n.length):"string"==r&&e in n)&&Vu(n[e],t)}function Oo(t,e){if(hp(t))return!1;var n=typeof t;return!("number"!=n&&"symbol"!=n&&"boolean"!=n&&null!=t&&!pc(t))||(Te.test(t)||!Me.test(t)||null!=e&&t in rl(e))}function Io(t){var e=typeof t;return"string"==e||"number"==e||"symbol"==e||"boolean"==e?"__proto__"!==t:null===t}function Do(t){var e=vo(t),r=n[e];if("function"!=typeof r||!(e in y.prototype))return!1;if(t===r)return!0;var i=bf(r);return!!i&&t===i[0]}function Ro(t){return!!dl&&dl in t}function Lo(t){var e=t&&t.constructor;return t===("function"==typeof e&&e.prototype||sl)}function Uo(t){return t===t&&!tc(t)}function Fo(t,e){return function(n){return null!=n&&(n[t]===e&&(e!==nt||t in rl(n)))}}function jo(t,e){var n=t[1],r=e[1],i=n|r,o=i<(dt|vt|xt),a=r==xt&&n==mt||r==xt&&n==wt&&t[7].length<=e[8]||r==(xt|wt)&&e[7].length<=e[8]&&n==mt;if(!o&&!a)return t;r&dt&&(t[2]=e[2],i|=n&dt?0:gt);var u=e[3];if(u){var c=t[3];t[3]=c?Ai(c,u,e[4]):u,t[4]=c?Y(t[3],ct):e[4]}return u=e[5],u&&(c=t[5],t[5]=c?Pi(c,u,e[6]):u,t[6]=c?Y(t[5],ct):e[6]),u=e[7],u&&(t[7]=u),r&xt&&(t[8]=null==t[8]?e[8]:Wl(t[8],e[8])),null==t[9]&&(t[9]=e[9]),t[0]=e[0],t[1]=i,t}function Bo(t){var e=[];if(null!=t)for(var n in rl(t))e.push(n);return e}function Vo(t){return vl.call(t)}function Wo(t,e,n){return e=Vl(e===nt?t.length-1:e,0),function(){for(var r=arguments,i=-1,a=Vl(r.length-e,0),u=Zs(a);++i<a;)u[i]=r[e+i];i=-1;for(var c=Zs(e+1);++i<e;)c[i]=r[i];return c[e]=n(u),o(t,this,c)}}function zo(t,e){return e.length<2?t:hr(t,ii(e,0,-1))}function Ho(t,e){for(var n=t.length,r=Wl(e.length,n),i=Oi(t);r--;){var o=e[r];t[r]=Ao(o,n)?i[o]:nt}return t}function qo(t,e){if("__proto__"!=e)return t[e]}function Yo(t,e,n){var r=e+"";return Tf(t,So(r,Qo(Co(r),n)))}function Ko(t){var e=0,n=0;return function(){var r=zl(),i=Tt-(r-n);if(n=r,i>0){if(++e>=Mt)return arguments[0]}else e=0;return t.apply(nt,arguments)}}function Go(t,e){var n=-1,r=t.length,i=r-1;for(e=e===nt?r:e;++n<e;){var o=Xr(n,i),a=t[o];t[o]=t[n],t[n]=a}return t.length=e,t}function $o(t){if("string"==typeof t||pc(t))return t;var e=t+"";return"0"==e&&1/t==-At?"-0":e}function Xo(t){if(null!=t){try{return fl.call(t)}catch(t){}try{return t+""}catch(t){}}return""}function Qo(t,e){return u(Ut,function(n){var r="_."+n[0];e&n[1]&&!f(t,r)&&t.push(r)}),t.sort()}function Zo(t){if(t instanceof y)return t.clone();var e=new i(t.__wrapped__,t.__chain__);return e.__actions__=Oi(t.__actions__),e.__index__=t.__index__,e.__values__=t.__values__,e}function Jo(t,e,n){e=(n?Po(t,e,n):e===nt)?1:Vl(yc(e),0);var r=null==t?0:t.length;if(!r||e<1)return[];for(var i=0,o=0,a=Zs(Dl(r/e));i<r;)a[o++]=ii(t,i,i+=e);return a}function ta(t){for(var e=-1,n=null==t?0:t.length,r=0,i=[];++e<n;){var o=t[e];o&&(i[r++]=o)}return i}function ea(){var t=arguments.length;if(!t)return[];for(var e=Zs(t-1),n=arguments[0],r=t;r--;)e[r-1]=arguments[r];return d(hp(n)?Oi(n):[n],sr(e,1))}function na(t,e,n){var r=null==t?0:t.length;return r?(e=n||e===nt?1:yc(e),ii(t,e<0?0:e,r)):[]}function ra(t,e,n){var r=null==t?0:t.length;return r?(e=n||e===nt?1:yc(e),e=r-e,ii(t,0,e<0?0:e)):[]}function ia(t,e){return t&&t.length?di(t,mo(e,3),!0,!0):[]}function oa(t,e){return t&&t.length?di(t,mo(e,3),!0):[]}function aa(t,e,n,r){var i=null==t?0:t.length;return i?(n&&"number"!=typeof n&&Po(t,e,n)&&(n=0,r=i),ur(t,e,n,r)):[]}function ua(t,e,n){var r=null==t?0:t.length;if(!r)return-1;var i=null==n?0:yc(n);return i<0&&(i=Vl(r+i,0)),x(t,mo(e,3),i)}function ca(t,e,n){var r=null==t?0:t.length;if(!r)return-1;var i=r-1;return n!==nt&&(i=yc(n),i=n<0?Vl(r+i,0):Wl(i,r-1)),x(t,mo(e,3),i,!0)}function sa(t){return(null==t?0:t.length)?sr(t,1):[]}function la(t){return(null==t?0:t.length)?sr(t,At):[]}function fa(t,e){return(null==t?0:t.length)?(e=e===nt?1:yc(e),sr(t,e)):[]}function pa(t){for(var e=-1,n=null==t?0:t.length,r={};++e<n;){var i=t[e];r[i[0]]=i[1]}return r}function ha(t){return t&&t.length?t[0]:nt}function da(t,e,n){var r=null==t?0:t.length;if(!r)return-1;var i=null==n?0:yc(n);return i<0&&(i=Vl(r+i,0)),w(t,e,i)}function va(t){return(null==t?0:t.length)?ii(t,0,-1):[]}function ga(t,e){return null==t?"":jl.call(t,e)}function ma(t){var e=null==t?0:t.length;return e?t[e-1]:nt}function ya(t,e,n){var r=null==t?0:t.length;if(!r)return-1;var i=r;return n!==nt&&(i=yc(n),i=i<0?Vl(r+i,0):Wl(i,r-1)),e===e?X(t,e,i):x(t,k,i,!0)}function _a(t,e){return t&&t.length?zr(t,yc(e)):nt}function ba(t,e){return t&&t.length&&e&&e.length?Gr(t,e):t}function xa(t,e,n){return t&&t.length&&e&&e.length?Gr(t,e,mo(n,2)):t}function wa(t,e,n){return t&&t.length&&e&&e.length?Gr(t,e,nt,n):t}function Ca(t,e){var n=[];if(!t||!t.length)return n;var r=-1,i=[],o=t.length;for(e=mo(e,3);++r<o;){var a=t[r];e(a,r,t)&&(n.push(a),i.push(r))}return $r(t,i),n}function ka(t){return null==t?t:Yl.call(t)}function Ea(t,e,n){var r=null==t?0:t.length;return r?(n&&"number"!=typeof n&&Po(t,e,n)?(e=0,n=r):(e=null==e?0:yc(e),n=n===nt?r:yc(n)),ii(t,e,n)):[]}function Ma(t,e){return ai(t,e)}function Ta(t,e,n){return ui(t,e,mo(n,2))}function Sa(t,e){var n=null==t?0:t.length;if(n){var r=ai(t,e);if(r<n&&Vu(t[r],e))return r}return-1}function Na(t,e){return ai(t,e,!0)}function Aa(t,e,n){return ui(t,e,mo(n,2),!0)}function Pa(t,e){if(null==t?0:t.length){var n=ai(t,e,!0)-1;if(Vu(t[n],e))return n}return-1}function Oa(t){return t&&t.length?ci(t):[]}function Ia(t,e){return t&&t.length?ci(t,mo(e,2)):[]}function Da(t){var e=null==t?0:t.length;return e?ii(t,1,e):[]}function Ra(t,e,n){return t&&t.length?(e=n||e===nt?1:yc(e),ii(t,0,e<0?0:e)):[]}function La(t,e,n){var r=null==t?0:t.length;return r?(e=n||e===nt?1:yc(e),e=r-e,ii(t,e<0?0:e,r)):[]}function Ua(t,e){return t&&t.length?di(t,mo(e,3),!1,!0):[]}function Fa(t,e){return t&&t.length?di(t,mo(e,3)):[]}function ja(t){return t&&t.length?fi(t):[]}function Ba(t,e){return t&&t.length?fi(t,mo(e,2)):[]}function Va(t,e){return e="function"==typeof e?e:nt,t&&t.length?fi(t,nt,e):[]}function Wa(t){if(!t||!t.length)return[];var e=0;return t=l(t,function(t){if(zu(t))return e=Vl(t.length,e),!0}),P(e,function(e){return h(t,M(e))})}function za(t,e){if(!t||!t.length)return[];var n=Wa(t);return null==e?n:h(n,function(t){return o(e,nt,t)})}function Ha(t,e){return mi(t||[],e||[],Wn)}function qa(t,e){return mi(t||[],e||[],ni)}function Ya(t){var e=n(t);return e.__chain__=!0,e}function Ka(t,e){return e(t),t}function Ga(t,e){return e(t)}function $a(){return Ya(this)}function Xa(){return new i(this.value(),this.__chain__)}function Qa(){this.__values__===nt&&(this.__values__=gc(this.value()));var t=this.__index__>=this.__values__.length;return{done:t,value:t?nt:this.__values__[this.__index__++]}}function Za(){return this}function Ja(t){for(var e,n=this;n instanceof r;){var i=Zo(n);i.__index__=0,i.__values__=nt,e?o.__wrapped__=i:e=i;var o=i;n=n.__wrapped__}return o.__wrapped__=t,e}function tu(){var t=this.__wrapped__;if(t instanceof y){var e=t;return this.__actions__.length&&(e=new y(this)),e=e.reverse(),e.__actions__.push({func:Ga,args:[ka],thisArg:nt}),new i(e,this.__chain__)}return this.thru(ka)}function eu(){return vi(this.__wrapped__,this.__actions__)}function nu(t,e,n){var r=hp(t)?s:or;return n&&Po(t,e,n)&&(e=nt),r(t,mo(e,3))}function ru(t,e){return(hp(t)?l:cr)(t,mo(e,3))}function iu(t,e){return sr(lu(t,e),1)}function ou(t,e){return sr(lu(t,e),At)}function au(t,e,n){return n=n===nt?1:yc(n),sr(lu(t,e),n)}function uu(t,e){return(hp(t)?u:ff)(t,mo(e,3))}function cu(t,e){return(hp(t)?c:pf)(t,mo(e,3))}function su(t,e,n,r){t=Wu(t)?t:$c(t),n=n&&!r?yc(n):0;var i=t.length;return n<0&&(n=Vl(i+n,0)),fc(t)?n<=i&&t.indexOf(e,n)>-1:!!i&&w(t,e,n)>-1}function lu(t,e){return(hp(t)?h:Fr)(t,mo(e,3))}function fu(t,e,n,r){return null==t?[]:(hp(e)||(e=null==e?[]:[e]),n=r?nt:n,hp(n)||(n=null==n?[]:[n]),Hr(t,e,n))}function pu(t,e,n){var r=hp(t)?v:S,i=arguments.length<3;return r(t,mo(e,4),n,i,ff)}function hu(t,e,n){var r=hp(t)?g:S,i=arguments.length<3;return r(t,mo(e,4),n,i,pf)}function du(t,e){return(hp(t)?l:cr)(t,Su(mo(e,3)))}function vu(t){return(hp(t)?Pn:ti)(t)}function gu(t,e,n){return e=(n?Po(t,e,n):e===nt)?1:yc(e),(hp(t)?On:ei)(t,e)}function mu(t){return(hp(t)?Dn:ri)(t)}function yu(t){if(null==t)return 0;if(Wu(t))return fc(t)?Q(t):t.length;var e=Cf(t);return e==Kt||e==Jt?t.size:Rr(t).length}function _u(t,e,n){var r=hp(t)?m:oi;return n&&Po(t,e,n)&&(e=nt),r(t,mo(e,3))}function bu(t,e){if("function"!=typeof e)throw new al(ot);return t=yc(t),function(){if(--t<1)return e.apply(this,arguments)}}function xu(t,e,n){return e=n?nt:e,e=t&&null==e?t.length:e,io(t,xt,nt,nt,nt,nt,e)}function wu(t,e){var n;if("function"!=typeof e)throw new al(ot);return t=yc(t),function(){return--t>0&&(n=e.apply(this,arguments)),t<=1&&(e=nt),n}}function Cu(t,e,n){e=n?nt:e;var r=io(t,mt,nt,nt,nt,nt,nt,e);return r.placeholder=Cu.placeholder,r}function ku(t,e,n){e=n?nt:e;var r=io(t,yt,nt,nt,nt,nt,nt,e);return r.placeholder=ku.placeholder,r}function Eu(t,e,n){function r(e){var n=p,r=h;return p=h=nt,y=e,v=t.apply(r,n)}function i(t){return y=t,g=Mf(u,e),_?r(t):v}function o(t){var n=t-m,r=t-y,i=e-n;return b?Wl(i,d-r):i}function a(t){var n=t-m,r=t-y;return m===nt||n>=e||n<0||b&&r>=d}function u(){var t=ep();if(a(t))return c(t);g=Mf(u,o(t))}function c(t){return g=nt,x&&p?r(t):(p=h=nt,v)}function s(){g!==nt&&yf(g),y=0,p=m=h=g=nt}function l(){return g===nt?v:c(ep())}function f(){var t=ep(),n=a(t);if(p=arguments,h=this,m=t,n){if(g===nt)return i(m);if(b)return g=Mf(u,e),r(m)}return g===nt&&(g=Mf(u,e)),v}var p,h,d,v,g,m,y=0,_=!1,b=!1,x=!0;if("function"!=typeof t)throw new al(ot);return e=bc(e)||0,tc(n)&&(_=!!n.leading,b="maxWait"in n,d=b?Vl(bc(n.maxWait)||0,e):d,x="trailing"in n?!!n.trailing:x),f.cancel=s,f.flush=l,f}function Mu(t){return io(t,Ct)}function Tu(t,e){if("function"!=typeof t||null!=e&&"function"!=typeof e)throw new al(ot);var n=function(){var r=arguments,i=e?e.apply(this,r):r[0],o=n.cache;if(o.has(i))return o.get(i);var a=t.apply(this,r);return n.cache=o.set(i,a)||o,a};return n.cache=new(Tu.Cache||an),n}function Su(t){if("function"!=typeof t)throw new al(ot);return function(){var e=arguments;switch(e.length){case 0:return!t.call(this);case 1:return!t.call(this,e[0]);case 2:return!t.call(this,e[0],e[1]);case 3:return!t.call(this,e[0],e[1],e[2])}return!t.apply(this,e)}}function Nu(t){return wu(2,t)}function Au(t,e){if("function"!=typeof t)throw new al(ot);return e=e===nt?e:yc(e),Jr(t,e)}function Pu(t,e){if("function"!=typeof t)throw new al(ot);return e=null==e?0:Vl(yc(e),0),Jr(function(n){var r=n[e],i=xi(n,0,e);return r&&d(i,r),o(t,this,i)})}function Ou(t,e,n){var r=!0,i=!0;if("function"!=typeof t)throw new al(ot);return tc(n)&&(r="leading"in n?!!n.leading:r,i="trailing"in n?!!n.trailing:i),Eu(t,e,{leading:r,maxWait:e,trailing:i})}function Iu(t){return xu(t,1)}function Du(t,e){return up(_i(e),t)}function Ru(){if(!arguments.length)return[];var t=arguments[0];return hp(t)?t:[t]}function Lu(t){return tr(t,ft)}function Uu(t,e){return e="function"==typeof e?e:nt,tr(t,ft,e)}function Fu(t){return tr(t,st|ft)}function ju(t,e){return e="function"==typeof e?e:nt,tr(t,st|ft,e)}function Bu(t,e){return null==e||nr(t,e,Lc(e))}function Vu(t,e){return t===e||t!==t&&e!==e}function Wu(t){return null!=t&&Ju(t.length)&&!Qu(t)}function zu(t){return ec(t)&&Wu(t)}function Hu(t){return!0===t||!1===t||ec(t)&&vr(t)==Vt}function qu(t){return ec(t)&&1===t.nodeType&&!sc(t)}function Yu(t){if(null==t)return!0;if(Wu(t)&&(hp(t)||"string"==typeof t||"function"==typeof t.splice||vp(t)||bp(t)||pp(t)))return!t.length;var e=Cf(t);if(e==Kt||e==Jt)return!t.size;if(Lo(t))return!Rr(t).length;for(var n in t)if(pl.call(t,n))return!1;return!0}function Ku(t,e){return Mr(t,e)}function Gu(t,e,n){n="function"==typeof n?n:nt;var r=n?n(t,e):nt;return r===nt?Mr(t,e,nt,n):!!r}function $u(t){if(!ec(t))return!1;var e=vr(t);return e==Ht||e==zt||"string"==typeof t.message&&"string"==typeof t.name&&!sc(t)}function Xu(t){return"number"==typeof t&&Fl(t)}function Qu(t){if(!tc(t))return!1;var e=vr(t);return e==qt||e==Yt||e==Bt||e==Qt}function Zu(t){return"number"==typeof t&&t==yc(t)}function Ju(t){return"number"==typeof t&&t>-1&&t%1==0&&t<=Pt}function tc(t){var e=typeof t;return null!=t&&("object"==e||"function"==e)}function ec(t){return null!=t&&"object"==typeof t}function nc(t,e){return t===e||Nr(t,e,_o(e))}function rc(t,e,n){return n="function"==typeof n?n:nt,Nr(t,e,_o(e),n)}function ic(t){return cc(t)&&t!=+t}function oc(t){if(kf(t))throw new tl(it);return Ar(t)}function ac(t){return null===t}function uc(t){return null==t}function cc(t){return"number"==typeof t||ec(t)&&vr(t)==Gt}function sc(t){if(!ec(t)||vr(t)!=Xt)return!1;var e=Cl(t);if(null===e)return!0;var n=pl.call(e,"constructor")&&e.constructor;return"function"==typeof n&&n instanceof n&&fl.call(n)==gl}function lc(t){return Zu(t)&&t>=-Pt&&t<=Pt}function fc(t){return"string"==typeof t||!hp(t)&&ec(t)&&vr(t)==te}function pc(t){return"symbol"==typeof t||ec(t)&&vr(t)==ee}function hc(t){return t===nt}function dc(t){return ec(t)&&Cf(t)==re}function vc(t){return ec(t)&&vr(t)==ie}function gc(t){if(!t)return[];if(Wu(t))return fc(t)?Z(t):Oi(t);if(Sl&&t[Sl])return z(t[Sl]());var e=Cf(t);return(e==Kt?H:e==Jt?K:$c)(t)}function mc(t){if(!t)return 0===t?t:0;if((t=bc(t))===At||t===-At){return(t<0?-1:1)*Ot}return t===t?t:0}function yc(t){var e=mc(t),n=e%1;return e===e?n?e-n:e:0}function _c(t){return t?Jn(yc(t),0,Dt):0}function bc(t){if("number"==typeof t)return t;if(pc(t))return It;if(tc(t)){var e="function"==typeof t.valueOf?t.valueOf():t;t=tc(e)?e+"":e}if("string"!=typeof t)return 0===t?t:+t;t=t.replace(Pe,"");var n=We.test(t);return n||He.test(t)?Tn(t.slice(2),n?2:8):Ve.test(t)?It:+t}function xc(t){return Ii(t,Uc(t))}function wc(t){return t?Jn(yc(t),-Pt,Pt):0===t?t:0}function Cc(t){return null==t?"":li(t)}function kc(t,e){var n=lf(t);return null==e?n:$n(n,e)}function Ec(t,e){return b(t,mo(e,3),lr)}function Mc(t,e){return b(t,mo(e,3),fr)}function Tc(t,e){return null==t?t:hf(t,mo(e,3),Uc)}function Sc(t,e){return null==t?t:df(t,mo(e,3),Uc)}function Nc(t,e){return t&&lr(t,mo(e,3))}function Ac(t,e){return t&&fr(t,mo(e,3))}function Pc(t){return null==t?[]:pr(t,Lc(t))}function Oc(t){return null==t?[]:pr(t,Uc(t))}function Ic(t,e,n){var r=null==t?nt:hr(t,e);return r===nt?n:r}function Dc(t,e){return null!=t&&ko(t,e,mr)}function Rc(t,e){return null!=t&&ko(t,e,yr)}function Lc(t){return Wu(t)?Nn(t):Rr(t)}function Uc(t){return Wu(t)?Nn(t,!0):Lr(t)}function Fc(t,e){var n={};return e=mo(e,3),lr(t,function(t,r,i){Qn(n,e(t,r,i),t)}),n}function jc(t,e){var n={};return e=mo(e,3),lr(t,function(t,r,i){Qn(n,r,e(t,r,i))}),n}function Bc(t,e){return Vc(t,Su(mo(e)))}function Vc(t,e){if(null==t)return{};var n=h(ho(t),function(t){return[t]});return e=mo(e),Yr(t,n,function(t,n){return e(t,n[0])})}function Wc(t,e,n){e=bi(e,t);var r=-1,i=e.length;for(i||(i=1,t=nt);++r<i;){var o=null==t?nt:t[$o(e[r])];o===nt&&(r=i,o=n),t=Qu(o)?o.call(t):o}return t}function zc(t,e,n){return null==t?t:ni(t,e,n)}function Hc(t,e,n,r){return r="function"==typeof r?r:nt,null==t?t:ni(t,e,n,r)}function qc(t,e,n){var r=hp(t),i=r||vp(t)||bp(t);if(e=mo(e,4),null==n){var o=t&&t.constructor;n=i?r?new o:[]:tc(t)&&Qu(o)?lf(Cl(t)):{}}return(i?u:lr)(t,function(t,r,i){return e(n,t,r,i)}),n}function Yc(t,e){return null==t||pi(t,e)}function Kc(t,e,n){return null==t?t:hi(t,e,_i(n))}function Gc(t,e,n,r){return r="function"==typeof r?r:nt,null==t?t:hi(t,e,_i(n),r)}function $c(t){return null==t?[]:D(t,Lc(t))}function Xc(t){return null==t?[]:D(t,Uc(t))}function Qc(t,e,n){return n===nt&&(n=e,e=nt),n!==nt&&(n=bc(n),n=n===n?n:0),e!==nt&&(e=bc(e),e=e===e?e:0),Jn(bc(t),e,n)}function Zc(t,e,n){return e=mc(e),n===nt?(n=e,e=0):n=mc(n),t=bc(t),_r(t,e,n)}function Jc(t,e,n){if(n&&"boolean"!=typeof n&&Po(t,e,n)&&(e=n=nt),n===nt&&("boolean"==typeof e?(n=e,e=nt):"boolean"==typeof t&&(n=t,t=nt)),t===nt&&e===nt?(t=0,e=1):(t=mc(t),e===nt?(e=t,t=0):e=mc(e)),t>e){var r=t;t=e,e=r}if(n||t%1||e%1){var i=ql();return Wl(t+i*(e-t+Mn("1e-"+((i+"").length-1))),e)}return Xr(t,e)}function ts(t){return Yp(Cc(t).toLowerCase())}function es(t){return(t=Cc(t))&&t.replace(Ye,zn).replace(hn,"")}function ns(t,e,n){t=Cc(t),e=li(e);var r=t.length;n=n===nt?r:Jn(yc(n),0,r);var i=n;return(n-=e.length)>=0&&t.slice(n,i)==e}function rs(t){return t=Cc(t),t&&we.test(t)?t.replace(be,Hn):t}function is(t){return t=Cc(t),t&&Ae.test(t)?t.replace(Ne,"\\$&"):t}function os(t,e,n){t=Cc(t),e=yc(e);var r=e?Q(t):0;if(!e||r>=e)return t;var i=(e-r)/2;return Qi(Rl(i),n)+t+Qi(Dl(i),n)}function as(t,e,n){t=Cc(t),e=yc(e);var r=e?Q(t):0;return e&&r<e?t+Qi(e-r,n):t}function us(t,e,n){t=Cc(t),e=yc(e);var r=e?Q(t):0;return e&&r<e?Qi(e-r,n)+t:t}function cs(t,e,n){return n||null==e?e=0:e&&(e=+e),Hl(Cc(t).replace(Oe,""),e||0)}function ss(t,e,n){return e=(n?Po(t,e,n):e===nt)?1:yc(e),Zr(Cc(t),e)}function ls(){var t=arguments,e=Cc(t[0]);return t.length<3?e:e.replace(t[1],t[2])}function fs(t,e,n){return n&&"number"!=typeof n&&Po(t,e,n)&&(e=n=nt),(n=n===nt?Dt:n>>>0)?(t=Cc(t),t&&("string"==typeof e||null!=e&&!yp(e))&&!(e=li(e))&&V(t)?xi(Z(t),0,n):t.split(e,n)):[]}function ps(t,e,n){return t=Cc(t),n=null==n?0:Jn(yc(n),0,t.length),e=li(e),t.slice(n,n+e.length)==e}function hs(t,e,r){var i=n.templateSettings;r&&Po(t,e,r)&&(e=nt),t=Cc(t),e=Ep({},e,i,oo);var o,a,u=Ep({},e.imports,i.imports,oo),c=Lc(u),s=D(u,c),l=0,f=e.interpolate||Ke,p="__p += '",h=il((e.escape||Ke).source+"|"+f.source+"|"+(f===Ee?je:Ke).source+"|"+(e.evaluate||Ke).source+"|$","g"),d="//# sourceURL="+("sourceURL"in e?e.sourceURL:"lodash.templateSources["+ ++_n+"]")+"\n";t.replace(h,function(e,n,r,i,u,c){return r||(r=i),p+=t.slice(l,c).replace(Ge,j),n&&(o=!0,p+="' +\n__e("+n+") +\n'"),u&&(a=!0,p+="';\n"+u+";\n__p += '"),r&&(p+="' +\n((__t = ("+r+")) == null ? '' : __t) +\n'"),l=c+e.length,e}),p+="';\n";var v=e.variable;v||(p="with (obj) {\n"+p+"\n}\n"),p=(a?p.replace(ge,""):p).replace(me,"$1").replace(ye,"$1;"),p="function("+(v||"obj")+") {\n"+(v?"":"obj || (obj = {});\n")+"var __t, __p = ''"+(o?", __e = _.escape":"")+(a?", __j = Array.prototype.join;\nfunction print() { __p += __j.call(arguments, '') }\n":";\n")+p+"return __p\n}";var g=Kp(function(){return el(c,d+"return "+p).apply(nt,s)});if(g.source=p,$u(g))throw g;return g}function ds(t){return Cc(t).toLowerCase()}function vs(t){return Cc(t).toUpperCase()}function gs(t,e,n){if((t=Cc(t))&&(n||e===nt))return t.replace(Pe,"");if(!t||!(e=li(e)))return t;var r=Z(t),i=Z(e);return xi(r,L(r,i),U(r,i)+1).join("")}function ms(t,e,n){if((t=Cc(t))&&(n||e===nt))return t.replace(Ie,"");if(!t||!(e=li(e)))return t;var r=Z(t);return xi(r,0,U(r,Z(e))+1).join("")}function ys(t,e,n){if((t=Cc(t))&&(n||e===nt))return t.replace(Oe,"");if(!t||!(e=li(e)))return t;var r=Z(t);return xi(r,L(r,Z(e))).join("")}function _s(t,e){var n=kt,r=Et;if(tc(e)){var i="separator"in e?e.separator:i;n="length"in e?yc(e.length):n,r="omission"in e?li(e.omission):r}t=Cc(t);var o=t.length;if(V(t)){var a=Z(t);o=a.length}if(n>=o)return t;var u=n-Q(r);if(u<1)return r;var c=a?xi(a,0,u).join(""):t.slice(0,u);if(i===nt)return c+r;if(a&&(u+=c.length-u),yp(i)){if(t.slice(u).search(i)){var s,l=c;for(i.global||(i=il(i.source,Cc(Be.exec(i))+"g")),i.lastIndex=0;s=i.exec(l);)var f=s.index;c=c.slice(0,f===nt?u:f)}}else if(t.indexOf(li(i),u)!=u){var p=c.lastIndexOf(i);p>-1&&(c=c.slice(0,p))}return c+r}function bs(t){return t=Cc(t),t&&xe.test(t)?t.replace(_e,qn):t}function xs(t,e,n){return t=Cc(t),e=n?nt:e,e===nt?W(t)?et(t):_(t):t.match(e)||[]}function ws(t){var e=null==t?0:t.length,n=mo();return t=e?h(t,function(t){if("function"!=typeof t[1])throw new al(ot);return[n(t[0]),t[1]]}):[],Jr(function(n){for(var r=-1;++r<e;){var i=t[r];if(o(i[0],this,n))return o(i[1],this,n)}})}function Cs(t){return er(tr(t,st))}function ks(t){return function(){return t}}function Es(t,e){return null==t||t!==t?e:t}function Ms(t){return t}function Ts(t){return Dr("function"==typeof t?t:tr(t,st))}function Ss(t){return jr(tr(t,st))}function Ns(t,e){return Br(t,tr(e,st))}function As(t,e,n){var r=Lc(e),i=pr(e,r);null!=n||tc(e)&&(i.length||!r.length)||(n=e,e=t,t=this,i=pr(e,Lc(e)));var o=!(tc(n)&&"chain"in n&&!n.chain),a=Qu(t);return u(i,function(n){var r=e[n];t[n]=r,a&&(t.prototype[n]=function(){var e=this.__chain__;if(o||e){var n=t(this.__wrapped__);return(n.__actions__=Oi(this.__actions__)).push({func:r,args:arguments,thisArg:t}),n.__chain__=e,n}return r.apply(t,d([this.value()],arguments))})}),t}function Ps(){return An._===this&&(An._=ml),this}function Os(){}function Is(t){return t=yc(t),Jr(function(e){return zr(e,t)})}function Ds(t){return Oo(t)?M($o(t)):Kr(t)}function Rs(t){return function(e){return null==t?nt:hr(t,e)}}function Ls(){return[]}function Us(){return!1}function Fs(){return{}}function js(){return""}function Bs(){return!0}function Vs(t,e){if((t=yc(t))<1||t>Pt)return[];var n=Dt,r=Wl(t,Dt);e=mo(e),t-=Dt;for(var i=P(r,e);++n<t;)e(n);return i}function Ws(t){return hp(t)?h(t,$o):pc(t)?[t]:Oi(Sf(Cc(t)))}function zs(t){var e=++hl;return Cc(t)+e}function Hs(t){return t&&t.length?ar(t,Ms,gr):nt}function qs(t,e){return t&&t.length?ar(t,mo(e,2),gr):nt}function Ys(t){return E(t,Ms)}function Ks(t,e){return E(t,mo(e,2))}function Gs(t){return t&&t.length?ar(t,Ms,Ur):nt}function $s(t,e){return t&&t.length?ar(t,mo(e,2),Ur):nt}function Xs(t){return t&&t.length?A(t,Ms):0}function Qs(t,e){return t&&t.length?A(t,mo(e,2)):0}e=null==e?An:Yn.defaults(An.Object(),e,Yn.pick(An,yn));var Zs=e.Array,Js=e.Date,tl=e.Error,el=e.Function,nl=e.Math,rl=e.Object,il=e.RegExp,ol=e.String,al=e.TypeError,ul=Zs.prototype,cl=el.prototype,sl=rl.prototype,ll=e["__core-js_shared__"],fl=cl.toString,pl=sl.hasOwnProperty,hl=0,dl=function(){var t=/[^.]+$/.exec(ll&&ll.keys&&ll.keys.IE_PROTO||"");return t?"Symbol(src)_1."+t:""}(),vl=sl.toString,gl=fl.call(rl),ml=An._,yl=il("^"+fl.call(pl).replace(Ne,"\\$&").replace(/hasOwnProperty|(function).*?(?=\\\()| for .+?(?=\\\])/g,"$1.*?")+"$"),_l=In?e.Buffer:nt,bl=e.Symbol,xl=e.Uint8Array,wl=_l?_l.allocUnsafe:nt,Cl=q(rl.getPrototypeOf,rl),kl=rl.create,El=sl.propertyIsEnumerable,Ml=ul.splice,Tl=bl?bl.isConcatSpreadable:nt,Sl=bl?bl.iterator:nt,Nl=bl?bl.toStringTag:nt,Al=function(){try{var t=bo(rl,"defineProperty");return t({},"",{}),t}catch(t){}}(),Pl=e.clearTimeout!==An.clearTimeout&&e.clearTimeout,Ol=Js&&Js.now!==An.Date.now&&Js.now,Il=e.setTimeout!==An.setTimeout&&e.setTimeout,Dl=nl.ceil,Rl=nl.floor,Ll=rl.getOwnPropertySymbols,Ul=_l?_l.isBuffer:nt,Fl=e.isFinite,jl=ul.join,Bl=q(rl.keys,rl),Vl=nl.max,Wl=nl.min,zl=Js.now,Hl=e.parseInt,ql=nl.random,Yl=ul.reverse,Kl=bo(e,"DataView"),Gl=bo(e,"Map"),$l=bo(e,"Promise"),Xl=bo(e,"Set"),Ql=bo(e,"WeakMap"),Zl=bo(rl,"create"),Jl=Ql&&new Ql,tf={},ef=Xo(Kl),nf=Xo(Gl),rf=Xo($l),of=Xo(Xl),af=Xo(Ql),uf=bl?bl.prototype:nt,cf=uf?uf.valueOf:nt,sf=uf?uf.toString:nt,lf=function(){function t(){}return function(e){if(!tc(e))return{};if(kl)return kl(e);t.prototype=e;var n=new t;return t.prototype=nt,n}}();n.templateSettings={escape:Ce,evaluate:ke,interpolate:Ee,variable:"",imports:{_:n}},n.prototype=r.prototype,n.prototype.constructor=n,i.prototype=lf(r.prototype),i.prototype.constructor=i,y.prototype=lf(r.prototype),y.prototype.constructor=y,tt.prototype.clear=Ue,tt.prototype.delete=$e,tt.prototype.get=Xe,tt.prototype.has=Qe,tt.prototype.set=Ze,Je.prototype.clear=tn,Je.prototype.delete=en,Je.prototype.get=nn,Je.prototype.has=rn,Je.prototype.set=on,an.prototype.clear=un,an.prototype.delete=cn,an.prototype.get=sn,an.prototype.has=ln,an.prototype.set=fn,dn.prototype.add=dn.prototype.push=vn,dn.prototype.has=gn,mn.prototype.clear=wn,mn.prototype.delete=Cn,mn.prototype.get=kn,mn.prototype.has=En,mn.prototype.set=Sn;var ff=Fi(lr),pf=Fi(fr,!0),hf=ji(),df=ji(!0),vf=Jl?function(t,e){return Jl.set(t,e),t}:Ms,gf=Al?function(t,e){return Al(t,"toString",{configurable:!0,enumerable:!1,value:ks(e),writable:!0})}:Ms,mf=Jr,yf=Pl||function(t){return An.clearTimeout(t)},_f=Xl&&1/K(new Xl([,-0]))[1]==At?function(t){return new Xl(t)}:Os,bf=Jl?function(t){return Jl.get(t)}:Os,xf=Ll?function(t){return null==t?[]:(t=rl(t),l(Ll(t),function(e){return El.call(t,e)}))}:Ls,wf=Ll?function(t){for(var e=[];t;)d(e,xf(t)),t=Cl(t);return e}:Ls,Cf=vr;(Kl&&Cf(new Kl(new ArrayBuffer(1)))!=ae||Gl&&Cf(new Gl)!=Kt||$l&&"[object Promise]"!=Cf($l.resolve())||Xl&&Cf(new Xl)!=Jt||Ql&&Cf(new Ql)!=re)&&(Cf=function(t){var e=vr(t),n=e==Xt?t.constructor:nt,r=n?Xo(n):"";if(r)switch(r){case ef:return ae;case nf:return Kt;case rf:return"[object Promise]";case of:return Jt;case af:return re}return e});var kf=ll?Qu:Us,Ef=Ko(vf),Mf=Il||function(t,e){return An.setTimeout(t,e)},Tf=Ko(gf),Sf=function(t){var e=Tu(t,function(t){return n.size===ut&&n.clear(),t}),n=e.cache;return e}(function(t){var e=[];return 46===t.charCodeAt(0)&&e.push(""),t.replace(Se,function(t,n,r,i){e.push(r?i.replace(Fe,"$1"):n||t)}),e}),Nf=Jr(function(t,e){return zu(t)?ir(t,sr(e,1,zu,!0)):[]}),Af=Jr(function(t,e){var n=ma(e);return zu(n)&&(n=nt),zu(t)?ir(t,sr(e,1,zu,!0),mo(n,2)):[]}),Pf=Jr(function(t,e){var n=ma(e);return zu(n)&&(n=nt),zu(t)?ir(t,sr(e,1,zu,!0),nt,n):[]}),Of=Jr(function(t){var e=h(t,yi);return e.length&&e[0]===t[0]?br(e):[]}),If=Jr(function(t){var e=ma(t),n=h(t,yi);return e===ma(n)?e=nt:n.pop(),n.length&&n[0]===t[0]?br(n,mo(e,2)):[]}),Df=Jr(function(t){var e=ma(t),n=h(t,yi);return e="function"==typeof e?e:nt,e&&n.pop(),n.length&&n[0]===t[0]?br(n,nt,e):[]}),Rf=Jr(ba),Lf=fo(function(t,e){var n=null==t?0:t.length,r=Zn(t,e);return $r(t,h(e,function(t){return Ao(t,n)?+t:t}).sort(Si)),r}),Uf=Jr(function(t){return fi(sr(t,1,zu,!0))}),Ff=Jr(function(t){var e=ma(t);return zu(e)&&(e=nt),fi(sr(t,1,zu,!0),mo(e,2))}),jf=Jr(function(t){var e=ma(t);return e="function"==typeof e?e:nt,fi(sr(t,1,zu,!0),nt,e)}),Bf=Jr(function(t,e){return zu(t)?ir(t,e):[]}),Vf=Jr(function(t){return gi(l(t,zu))}),Wf=Jr(function(t){var e=ma(t);return zu(e)&&(e=nt),gi(l(t,zu),mo(e,2))}),zf=Jr(function(t){var e=ma(t);return e="function"==typeof e?e:nt,gi(l(t,zu),nt,e)}),Hf=Jr(Wa),qf=Jr(function(t){var e=t.length,n=e>1?t[e-1]:nt;return n="function"==typeof n?(t.pop(),n):nt,za(t,n)}),Yf=fo(function(t){var e=t.length,n=e?t[0]:0,r=this.__wrapped__,o=function(e){return Zn(e,t)};return!(e>1||this.__actions__.length)&&r instanceof y&&Ao(n)?(r=r.slice(n,+n+(e?1:0)),r.__actions__.push({func:Ga,args:[o],thisArg:nt}),new i(r,this.__chain__).thru(function(t){return e&&!t.length&&t.push(nt),t})):this.thru(o)}),Kf=Li(function(t,e,n){pl.call(t,n)?++t[n]:Qn(t,n,1)}),Gf=qi(ua),$f=qi(ca),Xf=Li(function(t,e,n){pl.call(t,n)?t[n].push(e):Qn(t,n,[e])}),Qf=Jr(function(t,e,n){var r=-1,i="function"==typeof e,a=Wu(t)?Zs(t.length):[];return ff(t,function(t){a[++r]=i?o(e,t,n):wr(t,e,n)}),a}),Zf=Li(function(t,e,n){Qn(t,n,e)}),Jf=Li(function(t,e,n){t[n?0:1].push(e)},function(){return[[],[]]}),tp=Jr(function(t,e){if(null==t)return[];var n=e.length;return n>1&&Po(t,e[0],e[1])?e=[]:n>2&&Po(e[0],e[1],e[2])&&(e=[e[0]]),Hr(t,sr(e,1),[])}),ep=Ol||function(){return An.Date.now()},np=Jr(function(t,e,n){var r=dt;if(n.length){var i=Y(n,go(np));r|=_t}return io(t,r,e,n,i)}),rp=Jr(function(t,e,n){var r=dt|vt;if(n.length){var i=Y(n,go(rp));r|=_t}return io(e,r,t,n,i)}),ip=Jr(function(t,e){return rr(t,1,e)}),op=Jr(function(t,e,n){return rr(t,bc(e)||0,n)});Tu.Cache=an;var ap=mf(function(t,e){e=1==e.length&&hp(e[0])?h(e[0],I(mo())):h(sr(e,1),I(mo()));var n=e.length;return Jr(function(r){for(var i=-1,a=Wl(r.length,n);++i<a;)r[i]=e[i].call(this,r[i]);return o(t,this,r)})}),up=Jr(function(t,e){var n=Y(e,go(up));return io(t,_t,nt,e,n)}),cp=Jr(function(t,e){var n=Y(e,go(cp));return io(t,bt,nt,e,n)}),sp=fo(function(t,e){return io(t,wt,nt,nt,nt,e)}),lp=to(gr),fp=to(function(t,e){return t>=e}),pp=Cr(function(){return arguments}())?Cr:function(t){return ec(t)&&pl.call(t,"callee")&&!El.call(t,"callee")},hp=Zs.isArray,dp=Ln?I(Ln):kr,vp=Ul||Us,gp=Un?I(Un):Er,mp=Fn?I(Fn):Sr,yp=jn?I(jn):Pr,_p=Bn?I(Bn):Or,bp=Vn?I(Vn):Ir,xp=to(Ur),wp=to(function(t,e){return t<=e}),Cp=Ui(function(t,e){if(Lo(e)||Wu(e))return void Ii(e,Lc(e),t);for(var n in e)pl.call(e,n)&&Wn(t,n,e[n])}),kp=Ui(function(t,e){Ii(e,Uc(e),t)}),Ep=Ui(function(t,e,n,r){Ii(e,Uc(e),t,r)}),Mp=Ui(function(t,e,n,r){Ii(e,Lc(e),t,r)}),Tp=fo(Zn),Sp=Jr(function(t,e){t=rl(t);var n=-1,r=e.length,i=r>2?e[2]:nt;for(i&&Po(e[0],e[1],i)&&(r=1);++n<r;)for(var o=e[n],a=Uc(o),u=-1,c=a.length;++u<c;){var s=a[u],l=t[s];(l===nt||Vu(l,sl[s])&&!pl.call(t,s))&&(t[s]=o[s])}return t}),Np=Jr(function(t){return t.push(nt,ao),o(Dp,nt,t)}),Ap=Gi(function(t,e,n){null!=e&&"function"!=typeof e.toString&&(e=vl.call(e)),t[e]=n},ks(Ms)),Pp=Gi(function(t,e,n){null!=e&&"function"!=typeof e.toString&&(e=vl.call(e)),pl.call(t,e)?t[e].push(n):t[e]=[n]},mo),Op=Jr(wr),Ip=Ui(function(t,e,n){Vr(t,e,n)}),Dp=Ui(function(t,e,n,r){Vr(t,e,n,r)}),Rp=fo(function(t,e){var n={};if(null==t)return n;var r=!1;e=h(e,function(e){return e=bi(e,t),r||(r=e.length>1),e}),Ii(t,ho(t),n),r&&(n=tr(n,st|lt|ft,uo));for(var i=e.length;i--;)pi(n,e[i]);return n}),Lp=fo(function(t,e){return null==t?{}:qr(t,e)}),Up=ro(Lc),Fp=ro(Uc),jp=Wi(function(t,e,n){return e=e.toLowerCase(),t+(n?ts(e):e)}),Bp=Wi(function(t,e,n){return t+(n?"-":"")+e.toLowerCase()}),Vp=Wi(function(t,e,n){return t+(n?" ":"")+e.toLowerCase()}),Wp=Vi("toLowerCase"),zp=Wi(function(t,e,n){return t+(n?"_":"")+e.toLowerCase()}),Hp=Wi(function(t,e,n){return t+(n?" ":"")+Yp(e)}),qp=Wi(function(t,e,n){return t+(n?" ":"")+e.toUpperCase()}),Yp=Vi("toUpperCase"),Kp=Jr(function(t,e){try{return o(t,nt,e)}catch(t){return $u(t)?t:new tl(t)}}),Gp=fo(function(t,e){return u(e,function(e){e=$o(e),Qn(t,e,np(t[e],t))}),t}),$p=Yi(),Xp=Yi(!0),Qp=Jr(function(t,e){return function(n){return wr(n,t,e)}}),Zp=Jr(function(t,e){return function(n){return wr(t,n,e)}}),Jp=Xi(h),th=Xi(s),eh=Xi(m),nh=Ji(),rh=Ji(!0),ih=$i(function(t,e){return t+e},0),oh=no("ceil"),ah=$i(function(t,e){return t/e},1),uh=no("floor"),ch=$i(function(t,e){return t*e},1),sh=no("round"),lh=$i(function(t,e){return t-e},0);return n.after=bu,n.ary=xu,n.assign=Cp,n.assignIn=kp,n.assignInWith=Ep,n.assignWith=Mp,n.at=Tp,n.before=wu,n.bind=np,n.bindAll=Gp,n.bindKey=rp,n.castArray=Ru,n.chain=Ya,n.chunk=Jo,n.compact=ta,n.concat=ea,n.cond=ws,n.conforms=Cs,n.constant=ks,n.countBy=Kf,n.create=kc,n.curry=Cu,n.curryRight=ku,n.debounce=Eu,n.defaults=Sp,n.defaultsDeep=Np,n.defer=ip,n.delay=op,n.difference=Nf,n.differenceBy=Af,n.differenceWith=Pf,n.drop=na,n.dropRight=ra,n.dropRightWhile=ia,n.dropWhile=oa,n.fill=aa,n.filter=ru,n.flatMap=iu,n.flatMapDeep=ou,n.flatMapDepth=au,n.flatten=sa,n.flattenDeep=la,n.flattenDepth=fa,n.flip=Mu,n.flow=$p,n.flowRight=Xp,n.fromPairs=pa,n.functions=Pc,n.functionsIn=Oc,n.groupBy=Xf,n.initial=va,n.intersection=Of,n.intersectionBy=If,n.intersectionWith=Df,n.invert=Ap,n.invertBy=Pp,n.invokeMap=Qf,n.iteratee=Ts,n.keyBy=Zf,n.keys=Lc,n.keysIn=Uc,n.map=lu,n.mapKeys=Fc,n.mapValues=jc,n.matches=Ss,n.matchesProperty=Ns,n.memoize=Tu,n.merge=Ip,n.mergeWith=Dp,n.method=Qp,n.methodOf=Zp,n.mixin=As,n.negate=Su,n.nthArg=Is,n.omit=Rp,n.omitBy=Bc,n.once=Nu,n.orderBy=fu,n.over=Jp,n.overArgs=ap,n.overEvery=th,n.overSome=eh,n.partial=up,n.partialRight=cp,n.partition=Jf,n.pick=Lp,n.pickBy=Vc,n.property=Ds,n.propertyOf=Rs,n.pull=Rf,n.pullAll=ba,n.pullAllBy=xa,n.pullAllWith=wa,n.pullAt=Lf,n.range=nh,n.rangeRight=rh,n.rearg=sp,n.reject=du,n.remove=Ca,n.rest=Au,n.reverse=ka,n.sampleSize=gu,n.set=zc,n.setWith=Hc,n.shuffle=mu,n.slice=Ea,n.sortBy=tp,n.sortedUniq=Oa,n.sortedUniqBy=Ia,n.split=fs,n.spread=Pu,n.tail=Da,n.take=Ra,n.takeRight=La,n.takeRightWhile=Ua,n.takeWhile=Fa,n.tap=Ka,n.throttle=Ou,n.thru=Ga,n.toArray=gc,n.toPairs=Up,n.toPairsIn=Fp,n.toPath=Ws,n.toPlainObject=xc,n.transform=qc,n.unary=Iu,n.union=Uf,n.unionBy=Ff,n.unionWith=jf,n.uniq=ja,n.uniqBy=Ba,n.uniqWith=Va,n.unset=Yc,n.unzip=Wa,n.unzipWith=za,n.update=Kc,n.updateWith=Gc,n.values=$c,n.valuesIn=Xc,n.without=Bf,n.words=xs,n.wrap=Du,n.xor=Vf,n.xorBy=Wf,n.xorWith=zf,n.zip=Hf,n.zipObject=Ha,n.zipObjectDeep=qa,n.zipWith=qf,n.entries=Up,n.entriesIn=Fp,n.extend=kp,n.extendWith=Ep,As(n,n),n.add=ih,n.attempt=Kp,n.camelCase=jp,n.capitalize=ts,n.ceil=oh,n.clamp=Qc,n.clone=Lu,n.cloneDeep=Fu,n.cloneDeepWith=ju,n.cloneWith=Uu,n.conformsTo=Bu,n.deburr=es,n.defaultTo=Es,n.divide=ah,n.endsWith=ns,n.eq=Vu,n.escape=rs,n.escapeRegExp=is,n.every=nu,n.find=Gf,n.findIndex=ua,n.findKey=Ec,n.findLast=$f,n.findLastIndex=ca,n.findLastKey=Mc,n.floor=uh,n.forEach=uu,n.forEachRight=cu,n.forIn=Tc,n.forInRight=Sc,n.forOwn=Nc,n.forOwnRight=Ac,n.get=Ic,n.gt=lp,n.gte=fp,n.has=Dc,n.hasIn=Rc,n.head=ha,n.identity=Ms,n.includes=su,n.indexOf=da,n.inRange=Zc,n.invoke=Op,n.isArguments=pp,n.isArray=hp,n.isArrayBuffer=dp,n.isArrayLike=Wu,n.isArrayLikeObject=zu,n.isBoolean=Hu,n.isBuffer=vp,n.isDate=gp,n.isElement=qu,n.isEmpty=Yu,n.isEqual=Ku,n.isEqualWith=Gu,n.isError=$u,n.isFinite=Xu,n.isFunction=Qu,n.isInteger=Zu,n.isLength=Ju,n.isMap=mp,n.isMatch=nc,n.isMatchWith=rc,n.isNaN=ic,n.isNative=oc,n.isNil=uc,n.isNull=ac,n.isNumber=cc,n.isObject=tc,n.isObjectLike=ec,n.isPlainObject=sc,n.isRegExp=yp,n.isSafeInteger=lc,n.isSet=_p,n.isString=fc,n.isSymbol=pc,n.isTypedArray=bp,n.isUndefined=hc,n.isWeakMap=dc,n.isWeakSet=vc,n.join=ga,n.kebabCase=Bp,n.last=ma,n.lastIndexOf=ya,n.lowerCase=Vp,n.lowerFirst=Wp,n.lt=xp,n.lte=wp,n.max=Hs,n.maxBy=qs,n.mean=Ys,n.meanBy=Ks,n.min=Gs,n.minBy=$s,n.stubArray=Ls,n.stubFalse=Us,n.stubObject=Fs,n.stubString=js,n.stubTrue=Bs,n.multiply=ch,n.nth=_a,n.noConflict=Ps,n.noop=Os,n.now=ep,n.pad=os,n.padEnd=as,n.padStart=us,n.parseInt=cs,n.random=Jc,n.reduce=pu,n.reduceRight=hu,n.repeat=ss,n.replace=ls,n.result=Wc,n.round=sh,n.runInContext=t,n.sample=vu,n.size=yu,n.snakeCase=zp,n.some=_u,n.sortedIndex=Ma,n.sortedIndexBy=Ta,n.sortedIndexOf=Sa,n.sortedLastIndex=Na,n.sortedLastIndexBy=Aa,n.sortedLastIndexOf=Pa,n.startCase=Hp,n.startsWith=ps,n.subtract=lh,n.sum=Xs,n.sumBy=Qs,n.template=hs,n.times=Vs,n.toFinite=mc,n.toInteger=yc,n.toLength=_c,n.toLower=ds,n.toNumber=bc,n.toSafeInteger=wc,n.toString=Cc,n.toUpper=vs,n.trim=gs,n.trimEnd=ms,n.trimStart=ys,n.truncate=_s,n.unescape=bs,n.uniqueId=zs,n.upperCase=qp,n.upperFirst=Yp,n.each=uu,n.eachRight=cu,n.first=ha,As(n,function(){var t={};return lr(n,function(e,r){pl.call(n.prototype,r)||(t[r]=e)}),t}(),{chain:!1}),n.VERSION="4.17.11",u(["bind","bindKey","curry","curryRight","partial","partialRight"],function(t){n[t].placeholder=n}),u(["drop","take"],function(t,e){y.prototype[t]=function(n){n=n===nt?1:Vl(yc(n),0);var r=this.__filtered__&&!e?new y(this):this.clone();return r.__filtered__?r.__takeCount__=Wl(n,r.__takeCount__):r.__views__.push({size:Wl(n,Dt),type:t+(r.__dir__<0?"Right":"")}),r},y.prototype[t+"Right"]=function(e){return this.reverse()[t](e).reverse()}}),u(["filter","map","takeWhile"],function(t,e){var n=e+1,r=n==St||3==n;y.prototype[t]=function(t){var e=this.clone();return e.__iteratees__.push({iteratee:mo(t,3),type:n}),e.__filtered__=e.__filtered__||r,e}}),u(["head","last"],function(t,e){var n="take"+(e?"Right":"");y.prototype[t]=function(){return this[n](1).value()[0]}}),u(["initial","tail"],function(t,e){var n="drop"+(e?"":"Right");y.prototype[t]=function(){return this.__filtered__?new y(this):this[n](1)}}),y.prototype.compact=function(){return this.filter(Ms)},y.prototype.find=function(t){return this.filter(t).head()},y.prototype.findLast=function(t){return this.reverse().find(t)},y.prototype.invokeMap=Jr(function(t,e){return"function"==typeof t?new y(this):this.map(function(n){return wr(n,t,e)})}),y.prototype.reject=function(t){return this.filter(Su(mo(t)))},y.prototype.slice=function(t,e){t=yc(t);var n=this;return n.__filtered__&&(t>0||e<0)?new y(n):(t<0?n=n.takeRight(-t):t&&(n=n.drop(t)),e!==nt&&(e=yc(e),n=e<0?n.dropRight(-e):n.take(e-t)),n)},y.prototype.takeRightWhile=function(t){return this.reverse().takeWhile(t).reverse()},y.prototype.toArray=function(){return this.take(Dt)},lr(y.prototype,function(t,e){var r=/^(?:filter|find|map|reject)|While$/.test(e),o=/^(?:head|last)$/.test(e),a=n[o?"take"+("last"==e?"Right":""):e],u=o||/^find/.test(e);a&&(n.prototype[e]=function(){var e=this.__wrapped__,c=o?[1]:arguments,s=e instanceof y,l=c[0],f=s||hp(e),p=function(t){var e=a.apply(n,d([t],c));return o&&h?e[0]:e};f&&r&&"function"==typeof l&&1!=l.length&&(s=f=!1);var h=this.__chain__,v=!!this.__actions__.length,g=u&&!h,m=s&&!v;if(!u&&f){e=m?e:new y(this);var _=t.apply(e,c);return _.__actions__.push({func:Ga,args:[p],thisArg:nt}),new i(_,h)}return g&&m?t.apply(this,c):(_=this.thru(p),g?o?_.value()[0]:_.value():_)})}),u(["pop","push","shift","sort","splice","unshift"],function(t){var e=ul[t],r=/^(?:push|sort|unshift)$/.test(t)?"tap":"thru",i=/^(?:pop|shift)$/.test(t);n.prototype[t]=function(){var t=arguments;if(i&&!this.__chain__){var n=this.value();return e.apply(hp(n)?n:[],t)}return this[r](function(n){return e.apply(hp(n)?n:[],t)})}}),lr(y.prototype,function(t,e){var r=n[e];if(r){var i=r.name+"";(tf[i]||(tf[i]=[])).push({name:e,func:r})}}),tf[Ki(nt,vt).name]=[{name:"wrapper",func:nt}],y.prototype.clone=T,y.prototype.reverse=$,y.prototype.value=J,n.prototype.at=Yf,n.prototype.chain=$a,n.prototype.commit=Xa,n.prototype.next=Qa,n.prototype.plant=Ja,n.prototype.reverse=tu,n.prototype.toJSON=n.prototype.valueOf=n.prototype.value=eu,n.prototype.first=n.prototype.head,Sl&&(n.prototype[Sl]=Za),n}();An._=Yn,(i=function(){return Yn}.call(e,n,e,r))!==nt&&(r.exports=i)}).call(this)}).call(e,n(98),n(99)(t))},function(t,e,n){"use strict";var r={remove:function(t){t._reactInternalInstance=void 0},get:function(t){return t._reactInternalInstance},has:function(t){return void 0!==t._reactInternalInstance},set:function(t,e){t._reactInternalInstance=e}};t.exports=r},function(t,e,n){"use strict";function r(t){for(var e=arguments.length-1,n="Minified React error #"+t+"; visit http://facebook.github.io/react/docs/error-decoder.html?invariant="+t,r=0;r<e;r++)n+="&args[]="+encodeURIComponent(arguments[r+1]);n+=" for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";var i=new Error(n);throw i.name="Invariant Violation",i.framesToPop=1,i}t.exports=r},function(t,e,n){"use strict";t.exports=n(26)},function(t,e,n){"use strict";var r=n(63);e.a=function(t){return t=n.i(r.a)(Math.abs(t)),t?t[1]:NaN}},function(t,e,n){"use strict";e.a=function(t,e){return t=+t,e-=t,function(n){return t+e*n}}},function(t,e,n){"use strict";function r(t,e){return(e-=t=+t)?function(n){return(n-t)/e}:n.i(h.a)(e)}function i(t){return function(e,n){var r=t(e=+e,n=+n);return function(t){return t<=e?0:t>=n?1:r(t)}}}function o(t){return function(e,n){var r=t(e=+e,n=+n);return function(t){return t<=0?e:t>=1?n:r(t)}}}function a(t,e,n,r){var i=t[0],o=t[1],a=e[0],u=e[1];return o<i?(i=n(o,i),a=r(u,a)):(i=n(i,o),a=r(a,u)),function(t){return a(i(t))}}function u(t,e,r,i){var o=Math.min(t.length,e.length)-1,a=new Array(o),u=new Array(o),c=-1;for(t[o]<t[0]&&(t=t.slice().reverse(),e=e.slice().reverse());++c<o;)a[c]=r(t[c],t[c+1]),u[c]=i(e[c],e[c+1]);return function(e){var r=n.i(l.bisect)(t,e,1,o)-1;return u[r](a[r](e))}}function c(t,e){return e.domain(t.domain()).range(t.range()).interpolate(t.interpolate()).clamp(t.clamp())}function s(t,e){function n(){return s=Math.min(g.length,m.length)>2?u:a,l=h=null,c}function c(e){return(l||(l=s(g,m,_?i(t):t,y)))(+e)}var s,l,h,g=v,m=v,y=f.b,_=!1;return c.invert=function(t){return(h||(h=s(m,g,r,_?o(e):e)))(+t)},c.domain=function(t){return arguments.length?(g=p.a.call(t,d.a),n()):g.slice()},c.range=function(t){return arguments.length?(m=p.b.call(t),n()):m.slice()},c.rangeRound=function(t){return m=p.b.call(t),y=f.c,n()},c.clamp=function(t){return arguments.length?(_=!!t,n()):_},c.interpolate=function(t){return arguments.length?(y=t,n()):y},n()}e.b=r,e.c=c,e.a=s;var l=n(7),f=n(30),p=n(16),h=n(67),d=n(126),v=[0,1]},function(t,e,n){"use strict";function r(t){return function(){var e=this.ownerDocument,n=this.namespaceURI;return n===a.b&&e.documentElement.namespaceURI===a.b?e.createElement(t):e.createElementNS(n,t)}}function i(t){return function(){return this.ownerDocument.createElementNS(t.space,t.local)}}var o=n(68),a=n(69);e.a=function(t){var e=n.i(o.a)(t);return(e.local?i:r)(e)}},function(t,e,n){"use strict";e.a=function(t,e){var n=t.ownerSVGElement||t;if(n.createSVGPoint){var r=n.createSVGPoint();return r.x=e.clientX,r.y=e.clientY,r=r.matrixTransform(t.getScreenCTM().inverse()),[r.x,r.y]}var i=t.getBoundingClientRect();return[e.clientX-i.left-t.clientLeft,e.clientY-i.top-t.clientTop]}},function(t,e,n){"use strict";function r(t,e,n){t._context.bezierCurveTo((2*t._x0+t._x1)/3,(2*t._y0+t._y1)/3,(t._x0+2*t._x1)/3,(t._y0+2*t._y1)/3,(t._x0+4*t._x1+e)/6,(t._y0+4*t._y1+n)/6)}function i(t){this._context=t}e.c=r,e.b=i,i.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._y0=this._y1=NaN,this._point=0},lineEnd:function(){switch(this._point){case 3:r(this,this._x1,this._y1);case 2:this._context.lineTo(this._x1,this._y1)}(this._line||0!==this._line&&1===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1,this._line?this._context.lineTo(t,e):this._context.moveTo(t,e);break;case 1:this._point=2;break;case 2:this._point=3,this._context.lineTo((5*this._x0+this._x1)/6,(5*this._y0+this._y1)/6);default:r(this,t,e)}this._x0=this._x1,this._x1=t,this._y0=this._y1,this._y1=e}},e.a=function(t){return new i(t)}},function(t,e,n){"use strict";function r(t,e,n){t._context.bezierCurveTo(t._x1+t._k*(t._x2-t._x0),t._y1+t._k*(t._y2-t._y0),t._x2+t._k*(t._x1-e),t._y2+t._k*(t._y1-n),t._x2,t._y2)}function i(t,e){this._context=t,this._k=(1-e)/6}e.c=r,e.b=i,i.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._x2=this._y0=this._y1=this._y2=NaN,this._point=0},lineEnd:function(){switch(this._point){case 2:this._context.lineTo(this._x2,this._y2);break;case 3:r(this,this._x1,this._y1)}(this._line||0!==this._line&&1===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1,this._line?this._context.lineTo(t,e):this._context.moveTo(t,e);break;case 1:this._point=2,this._x1=t,this._y1=e;break;case 2:this._point=3;default:r(this,t,e)}this._x0=this._x1,this._x1=this._x2,this._x2=t,this._y0=this._y1,this._y1=this._y2,this._y2=e}},e.a=function t(e){function n(t){return new i(t,e)}return n.tension=function(e){return t(+e)},n}(0)},function(t,e,n){"use strict";function r(t){this._context=t}r.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._point=0},lineEnd:function(){(this._line||0!==this._line&&1===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1,this._line?this._context.lineTo(t,e):this._context.moveTo(t,e);break;case 1:this._point=2;default:this._context.lineTo(t,e)}}},e.a=function(t){return new r(t)}},function(t,e,n){"use strict";e.a=function(){}},function(t,e,n){"use strict";var r={};t.exports=r},function(t,e,n){"use strict";function r(t){return"topMouseUp"===t||"topTouchEnd"===t||"topTouchCancel"===t}function i(t){return"topMouseMove"===t||"topTouchMove"===t}function o(t){return"topMouseDown"===t||"topTouchStart"===t}function a(t,e,n,r){var i=t.type||"unknown-event";t.currentTarget=m.getNodeFromInstance(r),e?v.invokeGuardedCallbackWithCatch(i,n,t):v.invokeGuardedCallback(i,n,t),t.currentTarget=null}function u(t,e){var n=t._dispatchListeners,r=t._dispatchInstances;if(Array.isArray(n))for(var i=0;i<n.length&&!t.isPropagationStopped();i++)a(t,e,n[i],r[i]);else n&&a(t,e,n,r);t._dispatchListeners=null,t._dispatchInstances=null}function c(t){var e=t._dispatchListeners,n=t._dispatchInstances;if(Array.isArray(e)){for(var r=0;r<e.length&&!t.isPropagationStopped();r++)if(e[r](t,n[r]))return n[r]}else if(e&&e(t,n))return n;return null}function s(t){var e=c(t);return t._dispatchInstances=null,t._dispatchListeners=null,e}function l(t){var e=t._dispatchListeners,n=t._dispatchInstances;Array.isArray(e)&&d("103"),t.currentTarget=e?m.getNodeFromInstance(n):null;var r=e?e(t):null;return t.currentTarget=null,t._dispatchListeners=null,t._dispatchInstances=null,r}function f(t){return!!t._dispatchListeners}var p,h,d=n(1),v=n(88),g=(n(0),n(2),{injectComponentTree:function(t){p=t},injectTreeTraversal:function(t){h=t}}),m={isEndish:r,isMoveish:i,isStartish:o,executeDirectDispatch:l,executeDispatchesInOrder:u,executeDispatchesInOrderStopAtTrue:s,hasDispatches:f,getInstanceFromNode:function(t){return p.getInstanceFromNode(t)},getNodeFromInstance:function(t){return p.getNodeFromInstance(t)},isAncestor:function(t,e){return h.isAncestor(t,e)},getLowestCommonAncestor:function(t,e){return h.getLowestCommonAncestor(t,e)},getParentInstance:function(t){return h.getParentInstance(t)},traverseTwoPhase:function(t,e,n){return h.traverseTwoPhase(t,e,n)},traverseEnterLeave:function(t,e,n,r,i){return h.traverseEnterLeave(t,e,n,r,i)},injection:g};t.exports=m},function(t,e,n){"use strict";function r(t){return Object.prototype.hasOwnProperty.call(t,v)||(t[v]=h++,f[t[v]]={}),f[t[v]]}var i,o=n(3),a=n(84),u=n(374),c=n(90),s=n(406),l=n(95),f={},p=!1,h=0,d={topAbort:"abort",topAnimationEnd:s("animationend")||"animationend",topAnimationIteration:s("animationiteration")||"animationiteration",topAnimationStart:s("animationstart")||"animationstart",topBlur:"blur",topCanPlay:"canplay",topCanPlayThrough:"canplaythrough",topChange:"change",topClick:"click",topCompositionEnd:"compositionend",topCompositionStart:"compositionstart",topCompositionUpdate:"compositionupdate",topContextMenu:"contextmenu",topCopy:"copy",topCut:"cut",topDoubleClick:"dblclick",topDrag:"drag",topDragEnd:"dragend",topDragEnter:"dragenter",topDragExit:"dragexit",topDragLeave:"dragleave",topDragOver:"dragover",topDragStart:"dragstart",topDrop:"drop",topDurationChange:"durationchange",topEmptied:"emptied",topEncrypted:"encrypted",topEnded:"ended",topError:"error",topFocus:"focus",topInput:"input",topKeyDown:"keydown",topKeyPress:"keypress",topKeyUp:"keyup",topLoadedData:"loadeddata",topLoadedMetadata:"loadedmetadata",topLoadStart:"loadstart",topMouseDown:"mousedown",topMouseMove:"mousemove",topMouseOut:"mouseout",topMouseOver:"mouseover",topMouseUp:"mouseup",topPaste:"paste",topPause:"pause",topPlay:"play",topPlaying:"playing",topProgress:"progress",topRateChange:"ratechange",topScroll:"scroll",topSeeked:"seeked",topSeeking:"seeking",topSelectionChange:"selectionchange",topStalled:"stalled",topSuspend:"suspend",topTextInput:"textInput",topTimeUpdate:"timeupdate",topTouchCancel:"touchcancel",topTouchEnd:"touchend",topTouchMove:"touchmove",topTouchStart:"touchstart",topTransitionEnd:s("transitionend")||"transitionend",topVolumeChange:"volumechange",topWaiting:"waiting",topWheel:"wheel"},v="_reactListenersID"+String(Math.random()).slice(2),g=o({},u,{ReactEventListener:null,injection:{injectReactEventListener:function(t){t.setHandleTopLevel(g.handleTopLevel),g.ReactEventListener=t}},setEnabled:function(t){g.ReactEventListener&&g.ReactEventListener.setEnabled(t)},isEnabled:function(){return!(!g.ReactEventListener||!g.ReactEventListener.isEnabled())},listenTo:function(t,e){for(var n=e,i=r(n),o=a.registrationNameDependencies[t],u=0;u<o.length;u++){var c=o[u];i.hasOwnProperty(c)&&i[c]||("topWheel"===c?l("wheel")?g.ReactEventListener.trapBubbledEvent("topWheel","wheel",n):l("mousewheel")?g.ReactEventListener.trapBubbledEvent("topWheel","mousewheel",n):g.ReactEventListener.trapBubbledEvent("topWheel","DOMMouseScroll",n):"topScroll"===c?l("scroll",!0)?g.ReactEventListener.trapCapturedEvent("topScroll","scroll",n):g.ReactEventListener.trapBubbledEvent("topScroll","scroll",g.ReactEventListener.WINDOW_HANDLE):"topFocus"===c||"topBlur"===c?(l("focus",!0)?(g.ReactEventListener.trapCapturedEvent("topFocus","focus",n),g.ReactEventListener.trapCapturedEvent("topBlur","blur",n)):l("focusin")&&(g.ReactEventListener.trapBubbledEvent("topFocus","focusin",n),g.ReactEventListener.trapBubbledEvent("topBlur","focusout",n)),i.topBlur=!0,i.topFocus=!0):d.hasOwnProperty(c)&&g.ReactEventListener.trapBubbledEvent(c,d[c],n),i[c]=!0)}},trapBubbledEvent:function(t,e,n){return g.ReactEventListener.trapBubbledEvent(t,e,n)},trapCapturedEvent:function(t,e,n){return g.ReactEventListener.trapCapturedEvent(t,e,n)},supportsEventPageXY:function(){if(!document.createEvent)return!1;var t=document.createEvent("MouseEvent");return null!=t&&"pageX"in t},ensureScrollValueMonitoring:function(){if(void 0===i&&(i=g.supportsEventPageXY()),!i&&!p){var t=c.refreshScrollValues;g.ReactEventListener.monitorScrollValue(t),p=!0}}});t.exports=g},function(t,e,n){"use strict";function r(t,e,n,r){return i.call(this,t,e,n,r)}var i=n(25),o=n(90),a=n(93),u={screenX:null,screenY:null,clientX:null,clientY:null,ctrlKey:null,shiftKey:null,altKey:null,metaKey:null,getModifierState:a,button:function(t){var e=t.button;return"which"in t?e:2===e?2:4===e?1:0},buttons:null,relatedTarget:function(t){return t.relatedTarget||(t.fromElement===t.srcElement?t.toElement:t.fromElement)},pageX:function(t){return"pageX"in t?t.pageX:t.clientX+o.currentScrollLeft},pageY:function(t){return"pageY"in t?t.pageY:t.clientY+o.currentScrollTop}};i.augmentClass(r,u),t.exports=r},function(t,e,n){"use strict";var r=n(1),i=(n(0),{}),o={reinitializeTransaction:function(){this.transactionWrappers=this.getTransactionWrappers(),this.wrapperInitData?this.wrapperInitData.length=0:this.wrapperInitData=[],this._isInTransaction=!1},_isInTransaction:!1,getTransactionWrappers:null,isInTransaction:function(){return!!this._isInTransaction},perform:function(t,e,n,i,o,a,u,c){this.isInTransaction()&&r("27");var s,l;try{this._isInTransaction=!0,s=!0,this.initializeAll(0),l=t.call(e,n,i,o,a,u,c),s=!1}finally{try{if(s)try{this.closeAll(0)}catch(t){}else this.closeAll(0)}finally{this._isInTransaction=!1}}return l},initializeAll:function(t){for(var e=this.transactionWrappers,n=t;n<e.length;n++){var r=e[n];try{this.wrapperInitData[n]=i,this.wrapperInitData[n]=r.initialize?r.initialize.call(this):null}finally{if(this.wrapperInitData[n]===i)try{this.initializeAll(n+1)}catch(t){}}}},closeAll:function(t){this.isInTransaction()||r("28");for(var e=this.transactionWrappers,n=t;n<e.length;n++){var o,a=e[n],u=this.wrapperInitData[n];try{o=!0,u!==i&&a.close&&a.close.call(this,u),o=!1}finally{if(o)try{this.closeAll(n+1)}catch(t){}}}this.wrapperInitData.length=0}};t.exports=o},function(t,e,n){"use strict";function r(t){var e=""+t,n=o.exec(e);if(!n)return e;var r,i="",a=0,u=0;for(a=n.index;a<e.length;a++){switch(e.charCodeAt(a)){case 34:r="&quot;";break;case 38:r="&amp;";break;case 39:r="&#x27;";break;case 60:r="&lt;";break;case 62:r="&gt;";break;default:continue}u!==a&&(i+=e.substring(u,a)),u=a+1,i+=r}return u!==a?i+e.substring(u,a):i}function i(t){return"boolean"==typeof t||"number"==typeof t?""+t:r(t)}var o=/["'&<>]/;t.exports=i},function(t,e,n){"use strict";var r,i=n(6),o=n(83),a=/^[ \r\n\t\f]/,u=/<(!--|link|noscript|meta|script|style)[ \r\n\t\f\/>]/,c=n(91),s=c(function(t,e){if(t.namespaceURI!==o.svg||"innerHTML"in t)t.innerHTML=e;else{r=r||document.createElement("div"),r.innerHTML="<svg>"+e+"</svg>";for(var n=r.firstChild;n.firstChild;)t.appendChild(n.firstChild)}});if(i.canUseDOM){var l=document.createElement("div");l.innerHTML=" ",""===l.innerHTML&&(s=function(t,e){if(t.parentNode&&t.parentNode.replaceChild(t,t),a.test(e)||"<"===e[0]&&u.test(e)){t.innerHTML=String.fromCharCode(65279)+e;var n=t.firstChild;1===n.data.length?t.removeChild(n):n.deleteData(0,1)}else t.innerHTML=e}),l=null}t.exports=s},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.default={colors:{RdBu:["rgb(255, 13, 87)","rgb(30, 136, 229)"],GnPR:["rgb(24, 196, 93)","rgb(124, 82, 255)"],CyPU:["#0099C6","#990099"],PkYg:["#DD4477","#66AA00"],DrDb:["#B82E2E","#316395"],LpLb:["#994499","#22AA99"],YlDp:["#AAAA11","#6633CC"],OrId:["#E67300","#3E0099"]},gray:"#777"}},function(t,e,n){"use strict";var r=n(28);e.a=function(t,e,n){if(null==n&&(n=r.a),i=t.length){if((e=+e)<=0||i<2)return+n(t[0],0,t);if(e>=1)return+n(t[i-1],i-1,t);var i,o=(i-1)*e,a=Math.floor(o),u=+n(t[a],a,t);return u+(+n(t[a+1],a+1,t)-u)*(o-a)}}},function(t,e,n){"use strict";function r(){}function i(t,e){var n=new r;if(t instanceof r)t.each(function(t,e){n.set(e,t)});else if(Array.isArray(t)){var i,o=-1,a=t.length;if(null==e)for(;++o<a;)n.set(o,t[o]);else for(;++o<a;)n.set(e(i=t[o],o,t),i)}else if(t)for(var u in t)n.set(u,t[u]);return n}n.d(e,"b",function(){return o});var o="$";r.prototype=i.prototype={constructor:r,has:function(t){return o+t in this},get:function(t){return this[o+t]},set:function(t,e){return this[o+t]=e,this},remove:function(t){var e=o+t;return e in this&&delete this[e]},clear:function(){for(var t in this)t[0]===o&&delete this[t]},keys:function(){var t=[];for(var e in this)e[0]===o&&t.push(e.slice(1));return t},values:function(){var t=[];for(var e in this)e[0]===o&&t.push(this[e]);return t},entries:function(){var t=[];for(var e in this)e[0]===o&&t.push({key:e.slice(1),value:this[e]});return t},size:function(){var t=0;for(var e in this)e[0]===o&&++t;return t},empty:function(){for(var t in this)if(t[0]===o)return!1;return!0},each:function(t){for(var e in this)e[0]===o&&t(this[e],e.slice(1),this)}},e.a=i},function(t,e,n){"use strict";function r(){}function i(t){var e;return t=(t+"").trim().toLowerCase(),(e=x.exec(t))?(e=parseInt(e[1],16),new s(e>>8&15|e>>4&240,e>>4&15|240&e,(15&e)<<4|15&e,1)):(e=w.exec(t))?o(parseInt(e[1],16)):(e=C.exec(t))?new s(e[1],e[2],e[3],1):(e=k.exec(t))?new s(255*e[1]/100,255*e[2]/100,255*e[3]/100,1):(e=E.exec(t))?a(e[1],e[2],e[3],e[4]):(e=M.exec(t))?a(255*e[1]/100,255*e[2]/100,255*e[3]/100,e[4]):(e=T.exec(t))?l(e[1],e[2]/100,e[3]/100,1):(e=S.exec(t))?l(e[1],e[2]/100,e[3]/100,e[4]):N.hasOwnProperty(t)?o(N[t]):"transparent"===t?new s(NaN,NaN,NaN,0):null}function o(t){return new s(t>>16&255,t>>8&255,255&t,1)}function a(t,e,n,r){return r<=0&&(t=e=n=NaN),new s(t,e,n,r)}function u(t){return t instanceof r||(t=i(t)),t?(t=t.rgb(),new s(t.r,t.g,t.b,t.opacity)):new s}function c(t,e,n,r){return 1===arguments.length?u(t):new s(t,e,n,null==r?1:r)}function s(t,e,n,r){this.r=+t,this.g=+e,this.b=+n,this.opacity=+r}function l(t,e,n,r){return r<=0?t=e=n=NaN:n<=0||n>=1?t=e=NaN:e<=0&&(t=NaN),new h(t,e,n,r)}function f(t){if(t instanceof h)return new h(t.h,t.s,t.l,t.opacity);if(t instanceof r||(t=i(t)),!t)return new h;if(t instanceof h)return t;t=t.rgb();var e=t.r/255,n=t.g/255,o=t.b/255,a=Math.min(e,n,o),u=Math.max(e,n,o),c=NaN,s=u-a,l=(u+a)/2;return s?(c=e===u?(n-o)/s+6*(n<o):n===u?(o-e)/s+2:(e-n)/s+4,s/=l<.5?u+a:2-u-a,c*=60):s=l>0&&l<1?0:c,new h(c,s,l,t.opacity)}function p(t,e,n,r){return 1===arguments.length?f(t):new h(t,e,n,null==r?1:r)}function h(t,e,n,r){this.h=+t,this.s=+e,this.l=+n,this.opacity=+r}function d(t,e,n){return 255*(t<60?e+(n-e)*t/60:t<180?n:t<240?e+(n-e)*(240-t)/60:e)}e.f=r,n.d(e,"h",function(){return g}),n.d(e,"g",function(){return m}),e.a=i,e.e=u,e.b=c,e.d=s,e.c=p;var v=n(62),g=.7,m=1/g,y="\\s*([+-]?\\d+)\\s*",_="\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)\\s*",b="\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)%\\s*",x=/^#([0-9a-f]{3})$/,w=/^#([0-9a-f]{6})$/,C=new RegExp("^rgb\\("+[y,y,y]+"\\)$"),k=new RegExp("^rgb\\("+[b,b,b]+"\\)$"),E=new RegExp("^rgba\\("+[y,y,y,_]+"\\)$"),M=new RegExp("^rgba\\("+[b,b,b,_]+"\\)$"),T=new RegExp("^hsl\\("+[_,b,b]+"\\)$"),S=new RegExp("^hsla\\("+[_,b,b,_]+"\\)$"),N={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074};n.i(v.a)(r,i,{displayable:function(){return this.rgb().displayable()},toString:function(){return this.rgb()+""}}),n.i(v.a)(s,c,n.i(v.b)(r,{brighter:function(t){return t=null==t?m:Math.pow(m,t),new s(this.r*t,this.g*t,this.b*t,this.opacity)},darker:function(t){return t=null==t?g:Math.pow(g,t),new s(this.r*t,this.g*t,this.b*t,this.opacity)},rgb:function(){return this},displayable:function(){return 0<=this.r&&this.r<=255&&0<=this.g&&this.g<=255&&0<=this.b&&this.b<=255&&0<=this.opacity&&this.opacity<=1},toString:function(){var t=this.opacity;return t=isNaN(t)?1:Math.max(0,Math.min(1,t)),(1===t?"rgb(":"rgba(")+Math.max(0,Math.min(255,Math.round(this.r)||0))+", "+Math.max(0,Math.min(255,Math.round(this.g)||0))+", "+Math.max(0,Math.min(255,Math.round(this.b)||0))+(1===t?")":", "+t+")")}})),n.i(v.a)(h,p,n.i(v.b)(r,{brighter:function(t){return t=null==t?m:Math.pow(m,t),new h(this.h,this.s,this.l*t,this.opacity)},darker:function(t){return t=null==t?g:Math.pow(g,t),new h(this.h,this.s,this.l*t,this.opacity)},rgb:function(){var t=this.h%360+360*(this.h<0),e=isNaN(t)||isNaN(this.s)?0:this.s,n=this.l,r=n+(n<.5?n:1-n)*e,i=2*n-r;return new s(d(t>=240?t-240:t+120,i,r),d(t,i,r),d(t<120?t+240:t-120,i,r),this.opacity)},displayable:function(){return(0<=this.s&&this.s<=1||isNaN(this.s))&&0<=this.l&&this.l<=1&&0<=this.opacity&&this.opacity<=1}}))},function(t,e,n){"use strict";function r(t,e){var n=Object.create(t.prototype);for(var r in e)n[r]=e[r];return n}e.b=r,e.a=function(t,e,n){t.prototype=e.prototype=n,n.constructor=t}},function(t,e,n){"use strict";e.a=function(t,e){if((n=(t=e?t.toExponential(e-1):t.toExponential()).indexOf("e"))<0)return null;var n,r=t.slice(0,n);return[r.length>1?r[0]+r.slice(2):r,+t.slice(n+1)]}},function(t,e,n){"use strict";function r(t,e,n,r,i){var o=t*t,a=o*t;return((1-3*t+3*o-a)*e+(4-6*o+3*a)*n+(1+3*t+3*o-3*a)*r+a*i)/6}e.b=r,e.a=function(t){var e=t.length-1;return function(n){var i=n<=0?n=0:n>=1?(n=1,e-1):Math.floor(n*e),o=t[i],a=t[i+1],u=i>0?t[i-1]:2*o-a,c=i<e-1?t[i+2]:2*a-o;return r((n-i/e)*e,u,o,a,c)}}},function(t,e,n){"use strict";var r=n(10),i=n(123),o=n(118),a=n(121),u=n(43),c=n(122),s=n(124),l=n(120);e.a=function(t,e){var f,p=typeof e;return null==e||"boolean"===p?n.i(l.a)(e):("number"===p?u.a:"string"===p?(f=n.i(r.color)(e))?(e=f,i.a):s.a:e instanceof r.color?i.a:e instanceof Date?a.a:Array.isArray(e)?o.a:"function"!=typeof e.valueOf&&"function"!=typeof e.toString||isNaN(e)?c.a:u.a)(t,e)}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(239);n.d(e,"scaleBand",function(){return r.a}),n.d(e,"scalePoint",function(){return r.b});var i=n(245);n.d(e,"scaleIdentity",function(){return i.a});var o=n(34);n.d(e,"scaleLinear",function(){return o.a});var a=n(246);n.d(e,"scaleLog",function(){return a.a});var u=n(127);n.d(e,"scaleOrdinal",function(){return u.a}),n.d(e,"scaleImplicit",function(){return u.b});var c=n(247);n.d(e,"scalePow",function(){return c.a}),n.d(e,"scaleSqrt",function(){return c.b});var s=n(248);n.d(e,"scaleQuantile",function(){return s.a});var l=n(249);n.d(e,"scaleQuantize",function(){return l.a});var f=n(252);n.d(e,"scaleThreshold",function(){return f.a});var p=n(128);n.d(e,"scaleTime",function(){return p.a});var h=n(254);n.d(e,"scaleUtc",function(){return h.a});var d=n(240);n.d(e,"schemeCategory10",function(){return d.a});var v=n(242);n.d(e,"schemeCategory20b",function(){return v.a});var g=n(243);n.d(e,"schemeCategory20c",function(){return g.a});var m=n(241);n.d(e,"schemeCategory20",function(){return m.a});var y=n(244);n.d(e,"interpolateCubehelixDefault",function(){return y.a});var _=n(250);n.d(e,"interpolateRainbow",function(){return _.a}),n.d(e,"interpolateWarm",function(){return _.b}),n.d(e,"interpolateCool",function(){return _.c});var b=n(255);n.d(e,"interpolateViridis",function(){return b.a}),n.d(e,"interpolateMagma",function(){return b.b}),n.d(e,"interpolateInferno",function(){return b.c}),n.d(e,"interpolatePlasma",function(){return b.d});var x=n(251);n.d(e,"scaleSequential",function(){return x.a})},function(t,e,n){"use strict";e.a=function(t){return function(){return t}}},function(t,e,n){"use strict";var r=n(69);e.a=function(t){var e=t+="",n=e.indexOf(":");return n>=0&&"xmlns"!==(e=t.slice(0,n))&&(t=t.slice(n+1)),r.a.hasOwnProperty(e)?{space:r.a[e],local:t}:t}},function(t,e,n){"use strict";n.d(e,"b",function(){return r});var r="http://www.w3.org/1999/xhtml";e.a={svg:"http://www.w3.org/2000/svg",xhtml:r,xlink:"http://www.w3.org/1999/xlink",xml:"http://www.w3.org/XML/1998/namespace",xmlns:"http://www.w3.org/2000/xmlns/"}},function(t,e,n){"use strict";function r(t,e,n){return t=i(t,e,n),function(e){var n=e.relatedTarget;n&&(n===this||8&n.compareDocumentPosition(this))||t.call(this,e)}}function i(t,e,n){return function(r){var i=l;l=r;try{t.call(this,this.__data__,e,n)}finally{l=i}}}function o(t){return t.trim().split(/^|\s+/).map(function(t){var e="",n=t.indexOf(".");return n>=0&&(e=t.slice(n+1),t=t.slice(0,n)),{type:t,name:e}})}function a(t){return function(){var e=this.__on;if(e){for(var n,r=0,i=-1,o=e.length;r<o;++r)n=e[r],t.type&&n.type!==t.type||n.name!==t.name?e[++i]=n:this.removeEventListener(n.type,n.listener,n.capture);++i?e.length=i:delete this.__on}}}function u(t,e,n){var o=s.hasOwnProperty(t.type)?r:i;return function(r,i,a){var u,c=this.__on,s=o(e,i,a);if(c)for(var l=0,f=c.length;l<f;++l)if((u=c[l]).type===t.type&&u.name===t.name)return this.removeEventListener(u.type,u.listener,u.capture),this.addEventListener(u.type,u.listener=s,u.capture=n),void(u.value=e);this.addEventListener(t.type,s,n),u={type:t.type,name:t.name,value:e,listener:s,capture:n},c?c.push(u):this.__on=[u]}}function c(t,e,n,r){var i=l;t.sourceEvent=l,l=t;try{return e.apply(n,r)}finally{l=i}}n.d(e,"a",function(){return l}),e.b=c;var s={},l=null;if("undefined"!=typeof document){"onmouseenter"in document.documentElement||(s={mouseenter:"mouseover",mouseleave:"mouseout"})}e.c=function(t,e,n){var r,i,c=o(t+""),s=c.length;{if(!(arguments.length<2)){for(l=e?u:a,null==n&&(n=!1),r=0;r<s;++r)this.each(l(c[r],e,n));return this}var l=this.node().__on;if(l)for(var f,p=0,h=l.length;p<h;++p)for(r=0,f=l[p];r<s;++r)if((i=c[r]).type===f.type&&i.name===f.name)return f.value}}},function(t,e,n){"use strict";function r(){}e.a=function(t){return null==t?r:function(){return this.querySelector(t)}}},function(t,e,n){"use strict";var r=n(70);e.a=function(){for(var t,e=r.a;t=e.sourceEvent;)e=t;return e}},function(t,e,n){"use strict";e.a=function(t){return t.ownerDocument&&t.ownerDocument.defaultView||t.document&&t||t.defaultView}},function(t,e,n){"use strict";function r(t,e,n){var r=t._x1,i=t._y1,a=t._x2,u=t._y2;if(t._l01_a>o.a){var c=2*t._l01_2a+3*t._l01_a*t._l12_a+t._l12_2a,s=3*t._l01_a*(t._l01_a+t._l12_a);r=(r*c-t._x0*t._l12_2a+t._x2*t._l01_2a)/s,i=(i*c-t._y0*t._l12_2a+t._y2*t._l01_2a)/s}if(t._l23_a>o.a){var l=2*t._l23_2a+3*t._l23_a*t._l12_a+t._l12_2a,f=3*t._l23_a*(t._l23_a+t._l12_a);a=(a*l+t._x1*t._l23_2a-e*t._l12_2a)/f,u=(u*l+t._y1*t._l23_2a-n*t._l12_2a)/f}t._context.bezierCurveTo(r,i,a,u,t._x2,t._y2)}function i(t,e){this._context=t,this._alpha=e}e.b=r;var o=n(35),a=n(48);i.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._x2=this._y0=this._y1=this._y2=NaN,this._l01_a=this._l12_a=this._l23_a=this._l01_2a=this._l12_2a=this._l23_2a=this._point=0},lineEnd:function(){switch(this._point){case 2:this._context.lineTo(this._x2,this._y2);break;case 3:this.point(this._x2,this._y2)}(this._line||0!==this._line&&1===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){if(t=+t,e=+e,this._point){var n=this._x2-t,i=this._y2-e;this._l23_a=Math.sqrt(this._l23_2a=Math.pow(n*n+i*i,this._alpha))}switch(this._point){case 0:this._point=1,this._line?this._context.lineTo(t,e):this._context.moveTo(t,e);break;case 1:this._point=2;break;case 2:this._point=3;default:r(this,t,e)}this._l01_a=this._l12_a,this._l12_a=this._l23_a,this._l01_2a=this._l12_2a,this._l12_2a=this._l23_2a,this._x0=this._x1,this._x1=this._x2,this._x2=t,this._y0=this._y1,this._y1=this._y2,this._y2=e}},e.a=function t(e){function n(t){return e?new i(t,e):new a.b(t,0)}return n.alpha=function(e){return t(+e)},n}(.5)},function(t,e,n){"use strict";var r=n(32),i=n(17),o=n(49),a=n(77);e.a=function(){function t(t){var i,o,a,p=t.length,h=!1;for(null==s&&(f=l(a=n.i(r.a)())),i=0;i<=p;++i)!(i<p&&c(o=t[i],i,t))===h&&((h=!h)?f.lineStart():f.lineEnd()),h&&f.point(+e(o,i,t),+u(o,i,t));if(a)return f=null,a+""||null}var e=a.a,u=a.b,c=n.i(i.a)(!0),s=null,l=o.a,f=null;return t.x=function(r){return arguments.length?(e="function"==typeof r?r:n.i(i.a)(+r),t):e},t.y=function(e){return arguments.length?(u="function"==typeof e?e:n.i(i.a)(+e),t):u},t.defined=function(e){return arguments.length?(c="function"==typeof e?e:n.i(i.a)(!!e),t):c},t.curve=function(e){return arguments.length?(l=e,null!=s&&(f=l(s)),t):l},t.context=function(e){return arguments.length?(null==e?s=f=null:f=l(s=e),t):s},t}},function(t,e,n){"use strict";function r(t){for(var e,n=0,r=-1,i=t.length;++r<i;)(e=+t[r][1])&&(n+=e);return n}e.b=r;var i=n(37);e.a=function(t){var e=t.map(r);return n.i(i.a)(t).sort(function(t,n){return e[t]-e[n]})}},function(t,e,n){"use strict";function r(t){return t[0]}function i(t){return t[1]}e.a=r,e.b=i},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(79);n.d(e,"timeFormatDefaultLocale",function(){return r.a}),n.d(e,"timeFormat",function(){return r.b}),n.d(e,"timeParse",function(){return r.c}),n.d(e,"utcFormat",function(){return r.d}),n.d(e,"utcParse",function(){return r.e});var i=n(152);n.d(e,"timeFormatLocale",function(){return i.a});var o=n(151);n.d(e,"isoFormat",function(){return o.a});var a=n(314);n.d(e,"isoParse",function(){return a.a})},function(t,e,n){"use strict";function r(t){return i=n.i(s.a)(t),o=i.format,a=i.parse,u=i.utcFormat,c=i.utcParse,i}n.d(e,"b",function(){return o}),n.d(e,"c",function(){return a}),n.d(e,"d",function(){return u}),n.d(e,"e",function(){return c}),e.a=r;var i,o,a,u,c,s=n(152);r({dateTime:"%x, %X",date:"%-m/%-d/%Y",time:"%-I:%M:%S %p",periods:["AM","PM"],days:["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"],shortDays:["Sun","Mon","Tue","Wed","Thu","Fri","Sat"],months:["January","February","March","April","May","June","July","August","September","October","November","December"],shortMonths:["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]})},function(t,e,n){"use strict";var r=(n(5),n(317));n.d(e,"v",function(){return r.a}),n.d(e,"p",function(){return r.a});var i=n(320);n.d(e,"u",function(){return i.a}),n.d(e,"o",function(){return i.a});var o=n(318);n.d(e,"t",function(){return o.a});var a=n(316);n.d(e,"s",function(){return a.a});var u=n(315);n.d(e,"d",function(){return u.a});var c=n(327);n.d(e,"r",function(){return c.a}),n.d(e,"f",function(){return c.a}),n.d(e,"c",function(){return c.b}),n.d(e,"g",function(){return c.c});var s=n(319);n.d(e,"q",function(){return s.a});var l=n(328);n.d(e,"e",function(){return l.a});var f=n(323);n.d(e,"n",function(){return f.a});var p=n(322);n.d(e,"m",function(){return p.a});var h=n(321);n.d(e,"b",function(){return h.a});var d=n(325);n.d(e,"l",function(){return d.a}),n.d(e,"i",function(){return d.a}),n.d(e,"a",function(){return d.b}),n.d(e,"j",function(){return d.c});var v=n(324);n.d(e,"k",function(){return v.a});var g=n(326);n.d(e,"h",function(){return g.a})},function(t,e,n){"use strict";function r(t,e){return t===e?0!==t||0!==e||1/t==1/e:t!==t&&e!==e}function i(t,e){if(r(t,e))return!0;if("object"!=typeof t||null===t||"object"!=typeof e||null===e)return!1;var n=Object.keys(t),i=Object.keys(e);if(n.length!==i.length)return!1;for(var a=0;a<n.length;a++)if(!o.call(e,n[a])||!r(t[n[a]],e[n[a]]))return!1;return!0}var o=Object.prototype.hasOwnProperty;t.exports=i},function(t,e,n){"use strict";function r(t,e){return Array.isArray(e)&&(e=e[1]),e?e.nextSibling:t.firstChild}function i(t,e,n){l.insertTreeBefore(t,e,n)}function o(t,e,n){Array.isArray(e)?u(t,e[0],e[1],n):v(t,e,n)}function a(t,e){if(Array.isArray(e)){var n=e[1];e=e[0],c(t,e,n),t.removeChild(n)}t.removeChild(e)}function u(t,e,n,r){for(var i=e;;){var o=i.nextSibling;if(v(t,i,r),i===n)break;i=o}}function c(t,e,n){for(;;){var r=e.nextSibling;if(r===n)break;t.removeChild(r)}}function s(t,e,n){var r=t.parentNode,i=t.nextSibling;i===e?n&&v(r,document.createTextNode(n),i):n?(d(i,n),c(r,i,e)):c(r,t,e)}var l=n(20),f=n(350),p=(n(4),n(9),n(91)),h=n(57),d=n(176),v=p(function(t,e,n){t.insertBefore(e,n)}),g=f.dangerouslyReplaceNodeWithMarkup,m={dangerouslyReplaceNodeWithMarkup:g,replaceDelimitedText:s,processUpdates:function(t,e){for(var n=0;n<e.length;n++){var u=e[n];switch(u.type){case"INSERT_MARKUP":i(t,u.content,r(t,u.afterNode));break;case"MOVE_EXISTING":o(t,u.fromNode,r(t,u.afterNode));break;case"SET_MARKUP":h(t,u.content);break;case"TEXT_CONTENT":d(t,u.content);break;case"REMOVE_NODE":a(t,u.fromNode)}}}};t.exports=m},function(t,e,n){"use strict";var r={html:"http://www.w3.org/1999/xhtml",mathml:"http://www.w3.org/1998/Math/MathML",svg:"http://www.w3.org/2000/svg"};t.exports=r},function(t,e,n){"use strict";function r(){if(u)for(var t in c){var e=c[t],n=u.indexOf(t);if(n>-1||a("96",t),!s.plugins[n]){e.extractEvents||a("97",t),s.plugins[n]=e;var r=e.eventTypes;for(var o in r)i(r[o],e,o)||a("98",o,t)}}}function i(t,e,n){s.eventNameDispatchConfigs.hasOwnProperty(n)&&a("99",n),s.eventNameDispatchConfigs[n]=t;var r=t.phasedRegistrationNames;if(r){for(var i in r)if(r.hasOwnProperty(i)){var u=r[i];o(u,e,n)}return!0}return!!t.registrationName&&(o(t.registrationName,e,n),!0)}function o(t,e,n){s.registrationNameModules[t]&&a("100",t),s.registrationNameModules[t]=e,s.registrationNameDependencies[t]=e.eventTypes[n].dependencies}var a=n(1),u=(n(0),null),c={},s={plugins:[],eventNameDispatchConfigs:{},registrationNameModules:{},registrationNameDependencies:{},possibleRegistrationNames:null,injectEventPluginOrder:function(t){u&&a("101"),u=Array.prototype.slice.call(t),r()},injectEventPluginsByName:function(t){var e=!1;for(var n in t)if(t.hasOwnProperty(n)){var i=t[n];c.hasOwnProperty(n)&&c[n]===i||(c[n]&&a("102",n),c[n]=i,e=!0)}e&&r()},getPluginModuleForEvent:function(t){var e=t.dispatchConfig;if(e.registrationName)return s.registrationNameModules[e.registrationName]||null;if(void 0!==e.phasedRegistrationNames){var n=e.phasedRegistrationNames;for(var r in n)if(n.hasOwnProperty(r)){var i=s.registrationNameModules[n[r]];if(i)return i}}return null},_resetEventPlugins:function(){u=null;for(var t in c)c.hasOwnProperty(t)&&delete c[t];s.plugins.length=0;var e=s.eventNameDispatchConfigs;for(var n in e)e.hasOwnProperty(n)&&delete e[n];var r=s.registrationNameModules;for(var i in r)r.hasOwnProperty(i)&&delete r[i]}};t.exports=s},function(t,e,n){"use strict";function r(t){var e={"=":"=0",":":"=2"};return"$"+(""+t).replace(/[=:]/g,function(t){return e[t]})}function i(t){var e=/(=0|=2)/g,n={"=0":"=","=2":":"};return(""+("."===t[0]&&"$"===t[1]?t.substring(2):t.substring(1))).replace(e,function(t){return n[t]})}var o={escape:r,unescape:i};t.exports=o},function(t,e,n){"use strict";function r(t){null!=t.checkedLink&&null!=t.valueLink&&u("87")}function i(t){r(t),(null!=t.value||null!=t.onChange)&&u("88")}function o(t){r(t),(null!=t.checked||null!=t.onChange)&&u("89")}function a(t){if(t){var e=t.getName();if(e)return" Check the render method of `"+e+"`."}return""}var u=n(1),c=n(380),s=n(157),l=n(26),f=s(l.isValidElement),p=(n(0),n(2),{button:!0,checkbox:!0,image:!0,hidden:!0,radio:!0,reset:!0,submit:!0}),h={value:function(t,e,n){return!t[e]||p[t.type]||t.onChange||t.readOnly||t.disabled?null:new Error("You provided a `value` prop to a form field without an `onChange` handler. This will render a read-only field. If the field should be mutable use `defaultValue`. Otherwise, set either `onChange` or `readOnly`.")},checked:function(t,e,n){return!t[e]||t.onChange||t.readOnly||t.disabled?null:new Error("You provided a `checked` prop to a form field without an `onChange` handler. This will render a read-only field. If the field should be mutable use `defaultChecked`. Otherwise, set either `onChange` or `readOnly`.")},onChange:f.func},d={},v={checkPropTypes:function(t,e,n){for(var r in h){if(h.hasOwnProperty(r))var i=h[r](e,r,t,"prop",null,c);if(i instanceof Error&&!(i.message in d)){d[i.message]=!0;a(n)}}},getValue:function(t){return t.valueLink?(i(t),t.valueLink.value):t.value},getChecked:function(t){return t.checkedLink?(o(t),t.checkedLink.value):t.checked},executeOnChange:function(t,e){return t.valueLink?(i(t),t.valueLink.requestChange(e.target.value)):t.checkedLink?(o(t),t.checkedLink.requestChange(e.target.checked)):t.onChange?t.onChange.call(void 0,e):void 0}};t.exports=v},function(t,e,n){"use strict";var r=n(1),i=(n(0),!1),o={replaceNodeWithMarkup:null,processChildrenUpdates:null,injection:{injectEnvironment:function(t){i&&r("104"),o.replaceNodeWithMarkup=t.replaceNodeWithMarkup,o.processChildrenUpdates=t.processChildrenUpdates,i=!0}}};t.exports=o},function(t,e,n){"use strict";function r(t,e,n){try{e(n)}catch(t){null===i&&(i=t)}}var i=null,o={invokeGuardedCallback:r,invokeGuardedCallbackWithCatch:r,rethrowCaughtError:function(){if(i){var t=i;throw i=null,t}}};t.exports=o},function(t,e,n){"use strict";function r(t){c.enqueueUpdate(t)}function i(t){var e=typeof t;if("object"!==e)return e;var n=t.constructor&&t.constructor.name||e,r=Object.keys(t);return r.length>0&&r.length<20?n+" (keys: "+r.join(", ")+")":n}function o(t,e){var n=u.get(t);if(!n){return null}return n}var a=n(1),u=(n(15),n(39)),c=(n(9),n(12)),s=(n(0),n(2),{isMounted:function(t){var e=u.get(t);return!!e&&!!e._renderedComponent},enqueueCallback:function(t,e,n){s.validateCallback(e,n);var i=o(t);if(!i)return null;i._pendingCallbacks?i._pendingCallbacks.push(e):i._pendingCallbacks=[e],r(i)},enqueueCallbackInternal:function(t,e){t._pendingCallbacks?t._pendingCallbacks.push(e):t._pendingCallbacks=[e],r(t)},enqueueForceUpdate:function(t){var e=o(t,"forceUpdate");e&&(e._pendingForceUpdate=!0,r(e))},enqueueReplaceState:function(t,e,n){var i=o(t,"replaceState");i&&(i._pendingStateQueue=[e],i._pendingReplaceState=!0,void 0!==n&&null!==n&&(s.validateCallback(n,"replaceState"),i._pendingCallbacks?i._pendingCallbacks.push(n):i._pendingCallbacks=[n]),r(i))},enqueueSetState:function(t,e){var n=o(t,"setState");if(n){(n._pendingStateQueue||(n._pendingStateQueue=[])).push(e),r(n)}},enqueueElementInternal:function(t,e,n){t._pendingElement=e,t._context=n,r(t)},validateCallback:function(t,e){t&&"function"!=typeof t&&a("122",e,i(t))}});t.exports=s},function(t,e,n){"use strict";var r={currentScrollLeft:0,currentScrollTop:0,refreshScrollValues:function(t){r.currentScrollLeft=t.x,r.currentScrollTop=t.y}};t.exports=r},function(t,e,n){"use strict";var r=function(t){return"undefined"!=typeof MSApp&&MSApp.execUnsafeLocalFunction?function(e,n,r,i){MSApp.execUnsafeLocalFunction(function(){return t(e,n,r,i)})}:t};t.exports=r},function(t,e,n){"use strict";function r(t){var e,n=t.keyCode;return"charCode"in t?0===(e=t.charCode)&&13===n&&(e=13):e=n,e>=32||13===e?e:0}t.exports=r},function(t,e,n){"use strict";function r(t){var e=this,n=e.nativeEvent;if(n.getModifierState)return n.getModifierState(t);var r=o[t];return!!r&&!!n[r]}function i(t){return r}var o={Alt:"altKey",Control:"ctrlKey",Meta:"metaKey",Shift:"shiftKey"};t.exports=i},function(t,e,n){"use strict";function r(t){var e=t.target||t.srcElement||window;return e.correspondingUseElement&&(e=e.correspondingUseElement),3===e.nodeType?e.parentNode:e}t.exports=r},function(t,e,n){"use strict";/**
 * Checks if an event is supported in the current execution environment.
 *
 * NOTE: This will not work correctly for non-generic events such as `change`,
 * `reset`, `load`, `error`, and `select`.
 *
 * Borrows from Modernizr.
 *
 * @param {string} eventNameSuffix Event name, e.g. "click".
 * @param {?boolean} capture Check if the capture phase is supported.
 * @return {boolean} True if the event is supported.
 * @internal
 * @license Modernizr 3.0.0pre (Custom Build) | MIT
 */
function r(t,e){if(!o.canUseDOM||e&&!("addEventListener"in document))return!1;var n="on"+t,r=n in document;if(!r){var a=document.createElement("div");a.setAttribute(n,"return;"),r="function"==typeof a[n]}return!r&&i&&"wheel"===t&&(r=document.implementation.hasFeature("Events.wheel","3.0")),r}var i,o=n(6);o.canUseDOM&&(i=document.implementation&&document.implementation.hasFeature&&!0!==document.implementation.hasFeature("","")),t.exports=r},function(t,e,n){"use strict";function r(t,e){var n=null===t||!1===t,r=null===e||!1===e;if(n||r)return n===r;var i=typeof t,o=typeof e;return"string"===i||"number"===i?"string"===o||"number"===o:"object"===o&&t.type===e.type&&t.key===e.key}t.exports=r},function(t,e,n){"use strict";var r=(n(3),n(11)),i=(n(2),r);t.exports=i},function(t,e){var n;n=function(){return this}();try{n=n||Function("return this")()||(0,eval)("this")}catch(t){"object"==typeof window&&(n=window)}t.exports=n},function(t,e){t.exports=function(t){return t.webpackPolyfill||(t.deprecate=function(){},t.paths=[],t.children||(t.children=[]),Object.defineProperty(t,"loaded",{enumerable:!0,get:function(){return t.l}}),Object.defineProperty(t,"id",{enumerable:!0,get:function(){return t.i}}),t.webpackPolyfill=1),t}},function(t,e,n){"use strict";n.d(e,"b",function(){return i}),n.d(e,"a",function(){return o});var r=Array.prototype,i=r.slice,o=r.map},function(t,e,n){"use strict";n.d(e,"b",function(){return a}),n.d(e,"c",function(){return u});var r=n(19),i=n(102),o=n.i(i.a)(r.a),a=o.right,u=o.left;e.a=a},function(t,e,n){"use strict";function r(t){return function(e,r){return n.i(i.a)(t(e),r)}}var i=n(19);e.a=function(t){return 1===t.length&&(t=r(t)),{left:function(e,n,r,i){for(null==r&&(r=0),null==i&&(i=e.length);r<i;){var o=r+i>>>1;t(e[o],n)<0?r=o+1:i=o}return r},right:function(e,n,r,i){for(null==r&&(r=0),null==i&&(i=e.length);r<i;){var o=r+i>>>1;t(e[o],n)>0?i=o:r=o+1}return r}}}},function(t,e,n){"use strict";var r=n(111);e.a=function(t,e){var i=n.i(r.a)(t,e);return i?Math.sqrt(i):i}},function(t,e,n){"use strict";e.a=function(t,e){var n,r,i,o=t.length,a=-1;if(null==e){for(;++a<o;)if(null!=(n=t[a])&&n>=n)for(r=i=n;++a<o;)null!=(n=t[a])&&(r>n&&(r=n),i<n&&(i=n))}else for(;++a<o;)if(null!=(n=e(t[a],a,t))&&n>=n)for(r=i=n;++a<o;)null!=(n=e(t[a],a,t))&&(r>n&&(r=n),i<n&&(i=n));return[r,i]}},function(t,e,n){"use strict";e.a=function(t,e){var n,r,i=t.length,o=-1;if(null==e){for(;++o<i;)if(null!=(n=t[o])&&n>=n)for(r=n;++o<i;)null!=(n=t[o])&&r>n&&(r=n)}else for(;++o<i;)if(null!=(n=e(t[o],o,t))&&n>=n)for(r=n;++o<i;)null!=(n=e(t[o],o,t))&&r>n&&(r=n);return r}},function(t,e,n){"use strict";function r(t,e){return[t,e]}e.b=r,e.a=function(t,e){null==e&&(e=r);for(var n=0,i=t.length-1,o=t[0],a=new Array(i<0?0:i);n<i;)a[n]=e(o,o=t[++n]);return a}},function(t,e,n){"use strict";e.a=function(t,e,n){t=+t,e=+e,n=(i=arguments.length)<2?(e=t,t=0,1):i<3?1:+n;for(var r=-1,i=0|Math.max(0,Math.ceil((e-t)/n)),o=new Array(i);++r<i;)o[r]=t+r*n;return o}},function(t,e,n){"use strict";e.a=function(t){return Math.ceil(Math.log(t.length)/Math.LN2)+1}},function(t,e,n){"use strict";function r(t,e,n){var r=(e-t)/Math.max(0,n),i=Math.floor(Math.log(r)/Math.LN10),c=r/Math.pow(10,i);return i>=0?(c>=o?10:c>=a?5:c>=u?2:1)*Math.pow(10,i):-Math.pow(10,-i)/(c>=o?10:c>=a?5:c>=u?2:1)}function i(t,e,n){var r=Math.abs(e-t)/Math.max(0,n),i=Math.pow(10,Math.floor(Math.log(r)/Math.LN10)),c=r/i;return c>=o?i*=10:c>=a?i*=5:c>=u&&(i*=2),e<t?-i:i}e.b=r,e.c=i;var o=Math.sqrt(50),a=Math.sqrt(10),u=Math.sqrt(2);e.a=function(t,e,n){var i,o,a,u,c=-1;if(e=+e,t=+t,n=+n,t===e&&n>0)return[t];if((i=e<t)&&(o=t,t=e,e=o),0===(u=r(t,e,n))||!isFinite(u))return[];if(u>0)for(t=Math.ceil(t/u),e=Math.floor(e/u),a=new Array(o=Math.ceil(e-t+1));++c<o;)a[c]=(t+c)*u;else for(t=Math.floor(t*u),e=Math.ceil(e*u),a=new Array(o=Math.ceil(t-e+1));++c<o;)a[c]=(t-c)/u;return i&&a.reverse(),a}},function(t,e,n){"use strict";function r(t){return t.length}var i=n(105);e.a=function(t){if(!(u=t.length))return[];for(var e=-1,o=n.i(i.a)(t,r),a=new Array(o);++e<o;)for(var u,c=-1,s=a[e]=new Array(u);++c<u;)s[c]=t[c][e];return a}},function(t,e,n){"use strict";var r=n(28);e.a=function(t,e){var i,o,a=t.length,u=0,c=-1,s=0,l=0;if(null==e)for(;++c<a;)isNaN(i=n.i(r.a)(t[c]))||(o=i-s,s+=o/++u,l+=o*(i-s));else for(;++c<a;)isNaN(i=n.i(r.a)(e(t[c],c,t)))||(o=i-s,s+=o/++u,l+=o*(i-s));if(u>1)return l/(u-1)}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(209);n.d(e,"axisTop",function(){return r.a}),n.d(e,"axisRight",function(){return r.b}),n.d(e,"axisBottom",function(){return r.c}),n.d(e,"axisLeft",function(){return r.d})},function(t,e,n){"use strict";n.d(e,"b",function(){return r}),n.d(e,"a",function(){return i});var r=Math.PI/180,i=180/Math.PI},function(t,e,n){"use strict";n.d(e,"b",function(){return r});var r,i=n(63);e.a=function(t,e){var o=n.i(i.a)(t,e);if(!o)return t+"";var a=o[0],u=o[1],c=u-(r=3*Math.max(-8,Math.min(8,Math.floor(u/3))))+1,s=a.length;return c===s?a:c>s?a+new Array(c-s+1).join("0"):c>0?a.slice(0,c)+"."+a.slice(c):"0."+new Array(1-c).join("0")+n.i(i.a)(t,Math.max(0,e+c-1))[0]}},function(t,e,n){"use strict";function r(t){return new i(t)}function i(t){if(!(e=a.exec(t)))throw new Error("invalid format: "+t);var e,n=e[1]||" ",r=e[2]||">",i=e[3]||"-",u=e[4]||"",c=!!e[5],s=e[6]&&+e[6],l=!!e[7],f=e[8]&&+e[8].slice(1),p=e[9]||"";"n"===p?(l=!0,p="g"):o.a[p]||(p=""),(c||"0"===n&&"="===r)&&(c=!0,n="0",r="="),this.fill=n,this.align=r,this.sign=i,this.symbol=u,this.zero=c,this.width=s,this.comma=l,this.precision=f,this.type=p}e.a=r;var o=n(116),a=/^(?:(.)?([<>=^]))?([+\-\( ])?([$#])?(0)?(\d+)?(,)?(\.\d+)?([a-z%])?$/i;r.prototype=i.prototype,i.prototype.toString=function(){return this.fill+this.align+this.sign+this.symbol+(this.zero?"0":"")+(null==this.width?"":Math.max(1,0|this.width))+(this.comma?",":"")+(null==this.precision?"":"."+Math.max(0,0|this.precision))+this.type}},function(t,e,n){"use strict";var r=n(220),i=n(114),o=n(223);e.a={"":r.a,"%":function(t,e){return(100*t).toFixed(e)},b:function(t){return Math.round(t).toString(2)},c:function(t){return t+""},d:function(t){return Math.round(t).toString(10)},e:function(t,e){return t.toExponential(e)},f:function(t,e){return t.toFixed(e)},g:function(t,e){return t.toPrecision(e)},o:function(t){return Math.round(t).toString(8)},p:function(t,e){return n.i(o.a)(100*t,e)},r:o.a,s:i.a,X:function(t){return Math.round(t).toString(16).toUpperCase()},x:function(t){return Math.round(t).toString(16)}}},function(t,e,n){"use strict";var r=n(42),i=n(221),o=n(222),a=n(115),u=n(116),c=n(114),s=n(224),l=["y","z","a","f","p","n","µ","m","","k","M","G","T","P","E","Z","Y"];e.a=function(t){function e(t){function e(t){var e,n,a,u=x,s=w;if("c"===b)s=C(t)+s,t="";else{t=+t;var h=t<0;if(t=C(Math.abs(t),_),h&&0==+t&&(h=!1),u=(h?"("===o?o:"-":"-"===o||"("===o?"":o)+u,s=("s"===b?l[8+c.b/3]:"")+s+(h&&"("===o?")":""),k)for(e=-1,n=t.length;++e<n;)if(48>(a=t.charCodeAt(e))||a>57){s=(46===a?d+t.slice(e+1):t.slice(e))+s,t=t.slice(0,e);break}}y&&!f&&(t=p(t,1/0));var g=u.length+t.length+s.length,E=g<m?new Array(m-g+1).join(r):"";switch(y&&f&&(t=p(E+t,E.length?m-s.length:1/0),E=""),i){case"<":t=u+t+s+E;break;case"=":t=u+E+t+s;break;case"^":t=E.slice(0,g=E.length>>1)+u+t+s+E.slice(g);break;default:t=E+u+t+s}return v(t)}t=n.i(a.a)(t);var r=t.fill,i=t.align,o=t.sign,s=t.symbol,f=t.zero,m=t.width,y=t.comma,_=t.precision,b=t.type,x="$"===s?h[0]:"#"===s&&/[boxX]/.test(b)?"0"+b.toLowerCase():"",w="$"===s?h[1]:/[%p]/.test(b)?g:"",C=u.a[b],k=!b||/[defgprs%]/.test(b);return _=null==_?b?6:12:/[gprs]/.test(b)?Math.max(1,Math.min(21,_)):Math.max(0,Math.min(20,_)),e.toString=function(){return t+""},e}function f(t,i){var o=e((t=n.i(a.a)(t),t.type="f",t)),u=3*Math.max(-8,Math.min(8,Math.floor(n.i(r.a)(i)/3))),c=Math.pow(10,-u),s=l[8+u/3];return function(t){return o(c*t)+s}}var p=t.grouping&&t.thousands?n.i(i.a)(t.grouping,t.thousands):s.a,h=t.currency,d=t.decimal,v=t.numerals?n.i(o.a)(t.numerals):s.a,g=t.percent||"%";return{format:e,formatPrefix:f}}},function(t,e,n){"use strict";var r=n(65);e.a=function(t,e){var i,o=e?e.length:0,a=t?Math.min(o,t.length):0,u=new Array(a),c=new Array(o);for(i=0;i<a;++i)u[i]=n.i(r.a)(t[i],e[i]);for(;i<o;++i)c[i]=e[i];return function(t){for(i=0;i<a;++i)c[i]=u[i](t);return c}}},function(t,e,n){"use strict";var r=n(64);e.a=function(t){var e=t.length;return function(i){var o=Math.floor(((i%=1)<0?++i:i)*e),a=t[(o+e-1)%e],u=t[o%e],c=t[(o+1)%e],s=t[(o+2)%e];return n.i(r.b)((i-o/e)*e,a,u,c,s)}}},function(t,e,n){"use strict";e.a=function(t){return function(){return t}}},function(t,e,n){"use strict";e.a=function(t,e){var n=new Date;return t=+t,e-=t,function(r){return n.setTime(t+e*r),n}}},function(t,e,n){"use strict";var r=n(65);e.a=function(t,e){var i,o={},a={};null!==t&&"object"==typeof t||(t={}),null!==e&&"object"==typeof e||(e={});for(i in e)i in t?o[i]=n.i(r.a)(t[i],e[i]):a[i]=e[i];return function(t){for(i in o)a[i]=o[i](t);return a}}},function(t,e,n){"use strict";function r(t){return function(e){var r,o,a=e.length,u=new Array(a),c=new Array(a),s=new Array(a);for(r=0;r<a;++r)o=n.i(i.rgb)(e[r]),u[r]=o.r||0,c[r]=o.g||0,s[r]=o.b||0;return u=t(u),c=t(c),s=t(s),o.opacity=1,function(t){return o.r=u(t),o.g=c(t),o.b=s(t),o+""}}}var i=n(10),o=n(64),a=n(119),u=n(31);e.a=function t(e){function r(t,e){var r=o((t=n.i(i.rgb)(t)).r,(e=n.i(i.rgb)(e)).r),a=o(t.g,e.g),c=o(t.b,e.b),s=n.i(u.a)(t.opacity,e.opacity);return function(e){return t.r=r(e),t.g=a(e),t.b=c(e),t.opacity=s(e),t+""}}var o=n.i(u.c)(e);return r.gamma=t,r}(1);r(o.a),r(a.a)},function(t,e,n){"use strict";function r(t){return function(){return t}}function i(t){return function(e){return t(e)+""}}var o=n(43),a=/[-+]?(?:\d+\.?\d*|\.?\d+)(?:[eE][-+]?\d+)?/g,u=new RegExp(a.source,"g");e.a=function(t,e){var c,s,l,f=a.lastIndex=u.lastIndex=0,p=-1,h=[],d=[];for(t+="",e+="";(c=a.exec(t))&&(s=u.exec(e));)(l=s.index)>f&&(l=e.slice(f,l),h[p]?h[p]+=l:h[++p]=l),(c=c[0])===(s=s[0])?h[p]?h[p]+=s:h[++p]=s:(h[++p]=null,d.push({i:p,x:n.i(o.a)(c,s)})),f=u.lastIndex;return f<e.length&&(l=e.slice(f),h[p]?h[p]+=l:h[++p]=l),h.length<2?d[0]?i(d[0].x):r(e):(e=d.length,function(t){for(var n,r=0;r<e;++r)h[(n=d[r]).i]=n.x(t);return h.join("")})}},function(t,e,n){"use strict";e.a=function(t,e){t=t.slice();var n,r=0,i=t.length-1,o=t[r],a=t[i];return a<o&&(n=r,r=i,i=n,n=o,o=a,a=n),t[r]=e.floor(o),t[i]=e.ceil(a),t}},function(t,e,n){"use strict";e.a=function(t){return+t}},function(t,e,n){"use strict";function r(t){function e(e){var n=e+"",r=u.get(n);if(!r){if(s!==a)return s;u.set(n,r=c.push(e))}return t[(r-1)%t.length]}var u=n.i(i.a)(),c=[],s=a;return t=null==t?[]:o.b.call(t),e.domain=function(t){if(!arguments.length)return c.slice();c=[],u=n.i(i.a)();for(var r,o,a=-1,s=t.length;++a<s;)u.has(o=(r=t[a])+"")||u.set(o,c.push(r));return e},e.range=function(n){return arguments.length?(t=o.b.call(n),e):t.slice()},e.unknown=function(t){return arguments.length?(s=t,e):s},e.copy=function(){return r().domain(c).range(t).unknown(s)},e}n.d(e,"b",function(){return a}),e.a=r;var i=n(211),o=n(16),a={name:"implicit"}},function(t,e,n){"use strict";function r(t){return new Date(t)}function i(t){return t instanceof Date?+t:+new Date(+t)}function o(t,e,c,s,b,x,w,C,k){function E(n){return(w(n)<n?A:x(n)<n?P:b(n)<n?O:s(n)<n?I:e(n)<n?c(n)<n?D:R:t(n)<n?L:U)(n)}function M(e,r,i,o){if(null==e&&(e=10),"number"==typeof e){var u=Math.abs(i-r)/e,c=n.i(a.bisector)(function(t){return t[2]}).right(F,u);c===F.length?(o=n.i(a.tickStep)(r/_,i/_,e),e=t):c?(c=F[u/F[c-1][2]<F[c][2]/u?c-1:c],o=c[1],e=c[0]):(o=Math.max(n.i(a.tickStep)(r,i,e),1),e=C)}return null==o?e:e.every(o)}var T=n.i(f.a)(f.b,u.a),S=T.invert,N=T.domain,A=k(".%L"),P=k(":%S"),O=k("%I:%M"),I=k("%I %p"),D=k("%a %d"),R=k("%b %d"),L=k("%B"),U=k("%Y"),F=[[w,1,h],[w,5,5*h],[w,15,15*h],[w,30,30*h],[x,1,d],[x,5,5*d],[x,15,15*d],[x,30,30*d],[b,1,v],[b,3,3*v],[b,6,6*v],[b,12,12*v],[s,1,g],[s,2,2*g],[c,1,m],[e,1,y],[e,3,3*y],[t,1,_]];return T.invert=function(t){return new Date(S(t))},T.domain=function(t){return arguments.length?N(l.a.call(t,i)):N().map(r)},T.ticks=function(t,e){var n,r=N(),i=r[0],o=r[r.length-1],a=o<i;return a&&(n=i,i=o,o=n),n=M(t,i,o,e),n=n?n.range(i,o+1):[],a?n.reverse():n},T.tickFormat=function(t,e){return null==e?E:k(e)},T.nice=function(t,e){var r=N();return(t=M(t,r[0],r[r.length-1],e))?N(n.i(p.a)(r,t)):T},T.copy=function(){return n.i(f.c)(T,o(t,e,c,s,b,x,w,C,k))},T}e.b=o;var a=n(7),u=n(30),c=n(80),s=n(78),l=n(16),f=n(44),p=n(125),h=1e3,d=60*h,v=60*d,g=24*v,m=7*g,y=30*g,_=365*g;e.a=function(){return o(c.e,c.q,c.r,c.d,c.s,c.t,c.u,c.v,s.timeFormat).domain([new Date(2e3,0,1),new Date(2e3,0,2)])}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(257);n.d(e,"create",function(){return r.a});var i=n(45);n.d(e,"creator",function(){return i.a});var o=n(258);n.d(e,"local",function(){return o.a});var a=n(130);n.d(e,"matcher",function(){return a.a});var u=n(259);n.d(e,"mouse",function(){return u.a});var c=n(68);n.d(e,"namespace",function(){return c.a});var s=n(69);n.d(e,"namespaces",function(){return s.a});var l=n(46);n.d(e,"clientPoint",function(){return l.a});var f=n(131);n.d(e,"select",function(){return f.a});var p=n(260);n.d(e,"selectAll",function(){return p.a});var h=n(8);n.d(e,"selection",function(){return h.a});var d=n(71);n.d(e,"selector",function(){return d.a});var v=n(135);n.d(e,"selectorAll",function(){return v.a});var g=n(134);n.d(e,"style",function(){return g.a});var m=n(288);n.d(e,"touch",function(){return m.a});var y=n(289);n.d(e,"touches",function(){return y.a});var _=n(73);n.d(e,"window",function(){return _.a});var b=n(70);n.d(e,"event",function(){return b.a}),n.d(e,"customEvent",function(){return b.b})},function(t,e,n){"use strict";var r=function(t){return function(){return this.matches(t)}};if("undefined"!=typeof document){var i=document.documentElement;if(!i.matches){var o=i.webkitMatchesSelector||i.msMatchesSelector||i.mozMatchesSelector||i.oMatchesSelector;r=function(t){return function(){return o.call(this,t)}}}}e.a=r},function(t,e,n){"use strict";var r=n(8);e.a=function(t){return"string"==typeof t?new r.b([[document.querySelector(t)]],[document.documentElement]):new r.b([[t]],r.c)}},function(t,e,n){"use strict";function r(t,e){this.ownerDocument=t.ownerDocument,this.namespaceURI=t.namespaceURI,this._next=null,this._parent=t,this.__data__=e}e.b=r;var i=n(133),o=n(8);e.a=function(){return new o.b(this._enter||this._groups.map(i.a),this._parents)},r.prototype={constructor:r,appendChild:function(t){return this._parent.insertBefore(t,this._next)},insertBefore:function(t,e){return this._parent.insertBefore(t,e)},querySelector:function(t){return this._parent.querySelector(t)},querySelectorAll:function(t){return this._parent.querySelectorAll(t)}}},function(t,e,n){"use strict";e.a=function(t){return new Array(t.length)}},function(t,e,n){"use strict";function r(t){return function(){this.style.removeProperty(t)}}function i(t,e,n){return function(){this.style.setProperty(t,e,n)}}function o(t,e,n){return function(){var r=e.apply(this,arguments);null==r?this.style.removeProperty(t):this.style.setProperty(t,r,n)}}function a(t,e){return t.style.getPropertyValue(e)||n.i(u.a)(t).getComputedStyle(t,null).getPropertyValue(e)}e.a=a;var u=n(73);e.b=function(t,e,n){return arguments.length>1?this.each((null==e?r:"function"==typeof e?o:i)(t,e,null==n?"":n)):a(this.node(),t)}},function(t,e,n){"use strict";function r(){return[]}e.a=function(t){return null==t?r:function(){return this.querySelectorAll(t)}}},function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var r=n(290);n.d(e,"arc",function(){return r.a});var i=n(137);n.d(e,"area",function(){return i.a});var o=n(75);n.d(e,"line",function(){return o.a});var a=n(311);n.d(e,"pie",function(){return a.a});var u=n(291);n.d(e,"areaRadial",function(){return u.a}),n.d(e,"radialArea",function(){return u.a});var c=n(142);n.d(e,"lineRadial",function(){return c.a}),n.d(e,"radialLine",function(){return c.a});var s=n(143);n.d(e,"pointRadial",function(){return s.a});var l=n(303);n.d(e,"linkHorizontal",function(){return l.a}),n.d(e,"linkVertical",function(){return l.b}),n.d(e,"linkRadial",function(){return l.c});var f=n(313);n.d(e,"symbol",function(){return f.a}),n.d(e,"symbols",function(){return f.b});var p=n(144);n.d(e,"symbolCircle",function(){return p.a});var h=n(145);n.d(e,"symbolCross",function(){return h.a});var d=n(146);n.d(e,"symbolDiamond",function(){return d.a});var v=n(147);n.d(e,"symbolSquare",function(){return v.a});var g=n(148);n.d(e,"symbolStar",function(){return g.a});var m=n(149);n.d(e,"symbolTriangle",function(){return m.a});var y=n(150);n.d(e,"symbolWye",function(){return y.a});var _=n(292);n.d(e,"curveBasisClosed",function(){return _.a});var b=n(293);n.d(e,"curveBasisOpen",function(){return b.a});var x=n(47);n.d(e,"curveBasis",function(){return x.a});var w=n(294);n.d(e,"curveBundle",function(){return w.a});var C=n(139);n.d(e,"curveCardinalClosed",function(){return C.a});var k=n(140);n.d(e,"curveCardinalOpen",function(){return k.a});var E=n(48);n.d(e,"curveCardinal",function(){return E.a});var M=n(295);n.d(e,"curveCatmullRomClosed",function(){return M.a});var T=n(296);n.d(e,"curveCatmullRomOpen",function(){return T.a});var S=n(74);n.d(e,"curveCatmullRom",function(){return S.a});var N=n(297);n.d(e,"curveLinearClosed",function(){return N.a});var A=n(49);n.d(e,"curveLinear",function(){return A.a});var P=n(298);n.d(e,"curveMonotoneX",function(){return P.a}),n.d(e,"curveMonotoneY",function(){return P.b});var O=n(299);n.d(e,"curveNatural",function(){return O.a});var I=n(300);n.d(e,"curveStep",function(){return I.a}),n.d(e,"curveStepAfter",function(){return I.b}),n.d(e,"curveStepBefore",function(){return I.c});var D=n(312);n.d(e,"stack",function(){return D.a});var R=n(305);n.d(e,"stackOffsetExpand",function(){return R.a});var L=n(304);n.d(e,"stackOffsetDiverging",function(){return L.a});var U=n(36);n.d(e,"stackOffsetNone",function(){return U.a});var F=n(306);n.d(e,"stackOffsetSilhouette",function(){return F.a});var j=n(307);n.d(e,"stackOffsetWiggle",function(){return j.a});var B=n(76);n.d(e,"stackOrderAscending",function(){return B.a});var V=n(308);n.d(e,"stackOrderDescending",function(){return V.a});var W=n(309);n.d(e,"stackOrderInsideOut",function(){return W.a});var z=n(37);n.d(e,"stackOrderNone",function(){return z.a});var H=n(310);n.d(e,"stackOrderReverse",function(){return H.a})},function(t,e,n){"use strict";var r=n(32),i=n(17),o=n(49),a=n(75),u=n(77);e.a=function(){function t(t){var e,i,o,a,u,g=t.length,m=!1,y=new Array(g),_=new Array(g);for(null==h&&(v=d(u=n.i(r.a)())),e=0;e<=g;++e){if(!(e<g&&p(a=t[e],e,t))===m)if(m=!m)i=e,v.areaStart(),v.lineStart();else{for(v.lineEnd(),v.lineStart(),o=e-1;o>=i;--o)v.point(y[o],_[o]);v.lineEnd(),v.areaEnd()}m&&(y[e]=+c(a,e,t),_[e]=+l(a,e,t),v.point(s?+s(a,e,t):y[e],f?+f(a,e,t):_[e]))}if(u)return v=null,u+""||null}function e(){return n.i(a.a)().defined(p).curve(d).context(h)}var c=u.a,s=null,l=n.i(i.a)(0),f=u.b,p=n.i(i.a)(!0),h=null,d=o.a,v=null;return t.x=function(e){return arguments.length?(c="function"==typeof e?e:n.i(i.a)(+e),s=null,t):c},t.x0=function(e){return arguments.length?(c="function"==typeof e?e:n.i(i.a)(+e),t):c},t.x1=function(e){return arguments.length?(s=null==e?null:"function"==typeof e?e:n.i(i.a)(+e),t):s},t.y=function(e){return arguments.length?(l="function"==typeof e?e:n.i(i.a)(+e),f=null,t):l},t.y0=function(e){return arguments.length?(l="function"==typeof e?e:n.i(i.a)(+e),t):l},t.y1=function(e){return arguments.length?(f=null==e?null:"function"==typeof e?e:n.i(i.a)(+e),t):f},t.lineX0=t.lineY0=function(){return e().x(c).y(l)},t.lineY1=function(){return e().x(c).y(f)},t.lineX1=function(){return e().x(s).y(l)},t.defined=function(e){return arguments.length?(p="function"==typeof e?e:n.i(i.a)(!!e),t):p},t.curve=function(e){return arguments.length?(d=e,null!=h&&(v=d(h)),t):d},t.context=function(e){return arguments.length?(null==e?h=v=null:v=d(h=e),t):h},t}},function(t,e,n){"use strict";n.d(e,"a",function(){return r});var r=Array.prototype.slice},function(t,e,n){"use strict";function r(t,e){this._context=t,this._k=(1-e)/6}e.b=r;var i=n(50),o=n(48);r.prototype={areaStart:i.a,areaEnd:i.a,lineStart:function(){this._x0=this._x1=this._x2=this._x3=this._x4=this._x5=this._y0=this._y1=this._y2=this._y3=this._y4=this._y5=NaN,this._point=0},lineEnd:function(){switch(this._point){case 1:this._context.moveTo(this._x3,this._y3),this._context.closePath();break;case 2:this._context.lineTo(this._x3,this._y3),this._context.closePath();break;case 3:this.point(this._x3,this._y3),this.point(this._x4,this._y4),this.point(this._x5,this._y5)}},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1,this._x3=t,this._y3=e;break;case 1:this._point=2,this._context.moveTo(this._x4=t,this._y4=e);break;case 2:this._point=3,this._x5=t,this._y5=e;break;default:n.i(o.c)(this,t,e)}this._x0=this._x1,this._x1=this._x2,this._x2=t,this._y0=this._y1,this._y1=this._y2,this._y2=e}},e.a=function t(e){function n(t){return new r(t,e)}return n.tension=function(e){return t(+e)},n}(0)},function(t,e,n){"use strict";function r(t,e){this._context=t,this._k=(1-e)/6}e.b=r;var i=n(48);r.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._x2=this._y0=this._y1=this._y2=NaN,this._point=0},lineEnd:function(){(this._line||0!==this._line&&3===this._point)&&this._context.closePath(),this._line=1-this._line},point:function(t,e){switch(t=+t,e=+e,this._point){case 0:this._point=1;break;case 1:this._point=2;break;case 2:this._point=3,this._line?this._context.lineTo(this._x2,this._y2):this._context.moveTo(this._x2,this._y2);break;case 3:this._point=4;default:n.i(i.c)(this,t,e)}this._x0=this._x1,this._x1=this._x2,this._x2=t,this._y0=this._y1,this._y1=this._y2,this._y2=e}},e.a=function t(e){function n(t){return new r(t,e)}return n.tension=function(e){return t(+e)},n}(0)},function(t,e,n){"use strict";function r(t){this._curve=t}function i(t){function e(e){return new r(t(e))}return e._curve=t,e}n.d(e,"b",function(){return a}),e.a=i;var o=n(49),a=i(o.a);r.prototype={areaStart:function(){this._curve.areaStart()},areaEnd:function(){this._curve.areaEnd()},lineStart:function(){this._curve.lineStart()},lineEnd:function(){this._curve.lineEnd()},point:function(t,e){this._curve.point(e*Math.sin(t),e*-Math.cos(t))}}},function(t,e,n){"use strict";function r(t){var e=t.curve;return t.angle=t.x,delete t.x,t.radius=t.y,delete t.y,t.curve=function(t){return arguments.length?e(n.i(i.a)(t)):e()._curve},t}e.b=r;var i=n(141),o=n(75);e.a=function(){return r(n.i(o.a)().curve(i.b))}},function(t,e,n){"use strict";e.a=function(t,e){return[(e=+e)*Math.cos(t-=Math.PI/2),e*Math.sin(t)]}},function(t,e,n){"use strict";var r=n(35);e.a={draw:function(t,e){var n=Math.sqrt(e/r.b);t.moveTo(n,0),t.arc(0,0,n,0,r.c)}}},function(t,e,n){"use strict";e.a={draw:function(t,e){var n=Math.sqrt(e/5)/2;t.moveTo(-3*n,-n),t.lineTo(-n,-n),t.lineTo(-n,-3*n),t.lineTo(n,-3*n),t.lineTo(n,-n),t.lineTo(3*n,-n),t.lineTo(3*n,n),t.lineTo(n,n),t.lineTo(n,3*n),t.lineTo(-n,3*n),t.lineTo(-n,n),t.lineTo(-3*n,n),t.closePath()}}},function(t,e,n){"use strict";var r=Math.sqrt(1/3),i=2*r;e.a={draw:function(t,e){var n=Math.sqrt(e/i),o=n*r;t.moveTo(0,-n),t.lineTo(o,0),t.lineTo(0,n),t.lineTo(-o,0),t.closePath()}}},function(t,e,n){"use strict";e.a={draw:function(t,e){var n=Math.sqrt(e),r=-n/2;t.rect(r,r,n,n)}}},function(t,e,n){"use strict";var r=n(35),i=Math.sin(r.b/10)/Math.sin(7*r.b/10),o=Math.sin(r.c/10)*i,a=-Math.cos(r.c/10)*i;e.a={draw:function(t,e){var n=Math.sqrt(.8908130915292852*e),i=o*n,u=a*n;t.moveTo(0,-n),t.lineTo(i,u);for(var c=1;c<5;++c){var s=r.c*c/5,l=Math.cos(s),f=Math.sin(s);t.lineTo(f*n,-l*n),t.lineTo(l*i-f*u,f*i+l*u)}t.closePath()}}},function(t,e,n){"use strict";var r=Math.sqrt(3);e.a={draw:function(t,e){var n=-Math.sqrt(e/(3*r));t.moveTo(0,2*n),t.lineTo(-r*n,-n),t.lineTo(r*n,-n),t.closePath()}}},function(t,e,n){"use strict";var r=-.5,i=Math.sqrt(3)/2,o=1/Math.sqrt(12),a=3*(o/2+1);e.a={draw:function(t,e){var n=Math.sqrt(e/a),u=n/2,c=n*o,s=u,l=n*o+n,f=-s,p=l;t.moveTo(u,c),t.lineTo(s,l),t.lineTo(f,p),t.lineTo(r*u-i*c,i*u+r*c),t.lineTo(r*s-i*l,i*s+r*l),t.lineTo(r*f-i*p,i*f+r*p),t.lineTo(r*u+i*c,r*c-i*u),t.lineTo(r*s+i*l,r*l-i*s),t.lineTo(r*f+i*p,r*p-i*f),t.closePath()}}},function(t,e,n){"use strict";function r(t){return t.toISOString()}n.d(e,"b",function(){return o});var i=n(79),o="%Y-%m-%dT%H:%M:%S.%LZ",a=Date.prototype.toISOString?r:n.i(i.d)(o);e.a=a},function(t,e,n){"use strict";function r(t){if(0<=t.y&&t.y<100){var e=new Date(-1,t.m,t.d,t.H,t.M,t.S,t.L);return e.setFullYear(t.y),e}return new Date(t.y,t.m,t.d,t.H,t.M,t.S,t.L)}function i(t){if(0<=t.y&&t.y<100){var e=new Date(Date.UTC(-1,t.m,t.d,t.H,t.M,t.S,t.L));return e.setUTCFullYear(t.y),e}return new Date(Date.UTC(t.y,t.m,t.d,t.H,t.M,t.S,t.L))}function o(t){return{y:t,m:0,d:1,H:0,M:0,S:0,L:0}}function a(t){function e(t,e){return function(n){var r,i,o,a=[],u=-1,c=0,s=t.length;for(n instanceof Date||(n=new Date(+n));++u<s;)37===t.charCodeAt(u)&&(a.push(t.slice(c,u)),null!=(i=dt[r=t.charAt(++u)])?r=t.charAt(++u):i="e"===r?" ":"0",(o=e[r])&&(r=o(n,i)),a.push(r),c=u+1);return a.push(t.slice(c,u)),a.join("")}}function a(t,e){return function(r){var a,c,s=o(1900),l=u(s,t,r+="",0);if(l!=r.length)return null;if("Q"in s)return new Date(s.Q);if("p"in s&&(s.H=s.H%12+12*s.p),"V"in s){if(s.V<1||s.V>53)return null;"w"in s||(s.w=1),"Z"in s?(a=i(o(s.y)),c=a.getUTCDay(),a=c>4||0===c?ht.a.ceil(a):n.i(ht.a)(a),a=ht.b.offset(a,7*(s.V-1)),s.y=a.getUTCFullYear(),s.m=a.getUTCMonth(),s.d=a.getUTCDate()+(s.w+6)%7):(a=e(o(s.y)),c=a.getDay(),a=c>4||0===c?ht.c.ceil(a):n.i(ht.c)(a),a=ht.d.offset(a,7*(s.V-1)),s.y=a.getFullYear(),s.m=a.getMonth(),s.d=a.getDate()+(s.w+6)%7)}else("W"in s||"U"in s)&&("w"in s||(s.w="u"in s?s.u%7:"W"in s?1:0),c="Z"in s?i(o(s.y)).getUTCDay():e(o(s.y)).getDay(),s.m=0,s.d="W"in s?(s.w+6)%7+7*s.W-(c+5)%7:s.w+7*s.U-(c+6)%7);return"Z"in s?(s.H+=s.Z/100|0,s.M+=s.Z%100,i(s)):e(s)}}function u(t,e,n,r){for(var i,o,a=0,u=e.length,c=n.length;a<u;){if(r>=c)return-1;if(37===(i=e.charCodeAt(a++))){if(i=e.charAt(a++),!(o=Zt[i in dt?e.charAt(a++):i])||(r=o(t,n,r))<0)return-1}else if(i!=n.charCodeAt(r++))return-1}return r}function c(t,e,n){var r=Bt.exec(e.slice(n));return r?(t.p=Vt[r[0].toLowerCase()],n+r[0].length):-1}function vt(t,e,n){var r=Ht.exec(e.slice(n));return r?(t.w=qt[r[0].toLowerCase()],n+r[0].length):-1}function gt(t,e,n){var r=Wt.exec(e.slice(n));return r?(t.w=zt[r[0].toLowerCase()],n+r[0].length):-1}function mt(t,e,n){var r=Gt.exec(e.slice(n));return r?(t.m=$t[r[0].toLowerCase()],n+r[0].length):-1}function yt(t,e,n){var r=Yt.exec(e.slice(n));return r?(t.m=Kt[r[0].toLowerCase()],n+r[0].length):-1}function _t(t,e,n){return u(t,Ot,e,n)}function bt(t,e,n){return u(t,It,e,n)}function xt(t,e,n){return u(t,Dt,e,n)}function wt(t){return Ut[t.getDay()]}function Ct(t){return Lt[t.getDay()]}function kt(t){return jt[t.getMonth()]}function Et(t){return Ft[t.getMonth()]}function Mt(t){return Rt[+(t.getHours()>=12)]}function Tt(t){return Ut[t.getUTCDay()]}function St(t){return Lt[t.getUTCDay()]}function Nt(t){return jt[t.getUTCMonth()]}function At(t){return Ft[t.getUTCMonth()]}function Pt(t){return Rt[+(t.getUTCHours()>=12)]}var Ot=t.dateTime,It=t.date,Dt=t.time,Rt=t.periods,Lt=t.days,Ut=t.shortDays,Ft=t.months,jt=t.shortMonths,Bt=s(Rt),Vt=l(Rt),Wt=s(Lt),zt=l(Lt),Ht=s(Ut),qt=l(Ut),Yt=s(Ft),Kt=l(Ft),Gt=s(jt),$t=l(jt),Xt={a:wt,A:Ct,b:kt,B:Et,c:null,d:A,e:A,f:R,H:P,I:O,j:I,L:D,m:L,M:U,p:Mt,Q:ft,s:pt,S:F,u:j,U:B,V:V,w:W,W:z,x:null,X:null,y:H,Y:q,Z:Y,"%":lt},Qt={a:Tt,A:St,b:Nt,B:At,c:null,d:K,e:K,f:Z,H:G,I:$,j:X,L:Q,m:J,M:tt,p:Pt,Q:ft,s:pt,S:et,u:nt,U:rt,V:it,w:ot,W:at,x:null,X:null,y:ut,Y:ct,Z:st,"%":lt},Zt={a:vt,A:gt,b:mt,B:yt,c:_t,d:b,e:b,f:M,H:w,I:w,j:x,L:E,m:_,M:C,p:c,Q:S,s:N,S:k,u:p,U:h,V:d,w:f,W:v,x:bt,X:xt,y:m,Y:g,Z:y,"%":T};return Xt.x=e(It,Xt),Xt.X=e(Dt,Xt),Xt.c=e(Ot,Xt),Qt.x=e(It,Qt),Qt.X=e(Dt,Qt),Qt.c=e(Ot,Qt),{format:function(t){var n=e(t+="",Xt);return n.toString=function(){return t},n},parse:function(t){var e=a(t+="",r);return e.toString=function(){return t},e},utcFormat:function(t){var n=e(t+="",Qt);return n.toString=function(){return t},n},utcParse:function(t){var e=a(t,i);return e.toString=function(){return t},e}}}function u(t,e,n){var r=t<0?"-":"",i=(r?-t:t)+"",o=i.length;return r+(o<n?new Array(n-o+1).join(e)+i:i)}function c(t){return t.replace(mt,"\\$&")}function s(t){return new RegExp("^(?:"+t.map(c).join("|")+")","i")}function l(t){for(var e={},n=-1,r=t.length;++n<r;)e[t[n].toLowerCase()]=n;return e}function f(t,e,n){var r=vt.exec(e.slice(n,n+1));return r?(t.w=+r[0],n+r[0].length):-1}function p(t,e,n){var r=vt.exec(e.slice(n,n+1));return r?(t.u=+r[0],n+r[0].length):-1}function h(t,e,n){var r=vt.exec(e.slice(n,n+2));return r?(t.U=+r[0],n+r[0].length):-1}function d(t,e,n){var r=vt.exec(e.slice(n,n+2));return r?(t.V=+r[0],n+r[0].length):-1}function v(t,e,n){var r=vt.exec(e.slice(n,n+2));return r?(t.W=+r[0],n+r[0].length):-1}function g(t,e,n){var r=vt.exec(e.slice(n,n+4));return r?(t.y=+r[0],n+r[0].length):-1}function m(t,e,n){var r=vt.exec(e.slice(n,n+2));return r?(t.y=+r[0]+(+r[0]>68?1900:2e3),n+r[0].length):-1}function y(t,e,n){var r=/^(Z)|([+-]\d\d)(?::?(\d\d))?/.exec(e.slice(n,n+6));return r?(t.Z=r[1]?0:-(r[2]+(r[3]||"00")),n+r[0].length):-1}function _(t,e,n){var r=vt.exec(e.slice(n,n+2));return r?(t.m=r[0]-1,n+r[0].length):-1}function b(t,e,n){var r=vt.exec(e.slice(n,n+2));return r?(t.d=+r[0],n+r[0].length):-1}function x(t,e,n){var r=vt.exec(e.slice(n,n+3));return r?(t.m=0,t.d=+r[0],n+r[0].length):-1}function w(t,e,n){var r=vt.exec(e.slice(n,n+2));return r?(t.H=+r[0],n+r[0].length):-1}function C(t,e,n){var r=vt.exec(e.slice(n,n+2));return r?(t.M=+r[0],n+r[0].length):-1}function k(t,e,n){var r=vt.exec(e.slice(n,n+2));return r?(t.S=+r[0],n+r[0].length):-1}function E(t,e,n){var r=vt.exec(e.slice(n,n+3));return r?(t.L=+r[0],n+r[0].length):-1}function M(t,e,n){var r=vt.exec(e.slice(n,n+6));return r?(t.L=Math.floor(r[0]/1e3),n+r[0].length):-1}function T(t,e,n){var r=gt.exec(e.slice(n,n+1));return r?n+r[0].length:-1}function S(t,e,n){var r=vt.exec(e.slice(n));return r?(t.Q=+r[0],n+r[0].length):-1}function N(t,e,n){var r=vt.exec(e.slice(n));return r?(t.Q=1e3*+r[0],n+r[0].length):-1}function A(t,e){return u(t.getDate(),e,2)}function P(t,e){return u(t.getHours(),e,2)}function O(t,e){return u(t.getHours()%12||12,e,2)}function I(t,e){return u(1+ht.d.count(n.i(ht.e)(t),t),e,3)}function D(t,e){return u(t.getMilliseconds(),e,3)}function R(t,e){return D(t,e)+"000"}function L(t,e){return u(t.getMonth()+1,e,2)}function U(t,e){return u(t.getMinutes(),e,2)}function F(t,e){return u(t.getSeconds(),e,2)}function j(t){var e=t.getDay();return 0===e?7:e}function B(t,e){return u(ht.f.count(n.i(ht.e)(t),t),e,2)}function V(t,e){var r=t.getDay();return t=r>=4||0===r?n.i(ht.g)(t):ht.g.ceil(t),u(ht.g.count(n.i(ht.e)(t),t)+(4===n.i(ht.e)(t).getDay()),e,2)}function W(t){return t.getDay()}function z(t,e){return u(ht.c.count(n.i(ht.e)(t),t),e,2)}function H(t,e){return u(t.getFullYear()%100,e,2)}function q(t,e){return u(t.getFullYear()%1e4,e,4)}function Y(t){var e=t.getTimezoneOffset();return(e>0?"-":(e*=-1,"+"))+u(e/60|0,"0",2)+u(e%60,"0",2)}function K(t,e){return u(t.getUTCDate(),e,2)}function G(t,e){return u(t.getUTCHours(),e,2)}function $(t,e){return u(t.getUTCHours()%12||12,e,2)}function X(t,e){return u(1+ht.b.count(n.i(ht.h)(t),t),e,3)}function Q(t,e){return u(t.getUTCMilliseconds(),e,3)}function Z(t,e){return Q(t,e)+"000"}function J(t,e){return u(t.getUTCMonth()+1,e,2)}function tt(t,e){return u(t.getUTCMinutes(),e,2)}function et(t,e){return u(t.getUTCSeconds(),e,2)}function nt(t){var e=t.getUTCDay();return 0===e?7:e}function rt(t,e){return u(ht.i.count(n.i(ht.h)(t),t),e,2)}function it(t,e){var r=t.getUTCDay();return t=r>=4||0===r?n.i(ht.j)(t):ht.j.ceil(t),u(ht.j.count(n.i(ht.h)(t),t)+(4===n.i(ht.h)(t).getUTCDay()),e,2)}function ot(t){return t.getUTCDay()}function at(t,e){return u(ht.a.count(n.i(ht.h)(t),t),e,2)}function ut(t,e){return u(t.getUTCFullYear()%100,e,2)}function ct(t,e){return u(t.getUTCFullYear()%1e4,e,4)}function st(){return"+0000"}function lt(){return"%"}function ft(t){return+t}function pt(t){return Math.floor(+t/1e3)}e.a=a;var ht=n(80),dt={"-":"",_:" ",0:"0"},vt=/^\s*\d+/,gt=/^%/,mt=/[\\^$*+?|[\]().{}]/g},function(t,e,n){"use strict";var r=n(11),i={listen:function(t,e,n){return t.addEventListener?(t.addEventListener(e,n,!1),{remove:function(){t.removeEventListener(e,n,!1)}}):t.attachEvent?(t.attachEvent("on"+e,n),{remove:function(){t.detachEvent("on"+e,n)}}):void 0},capture:function(t,e,n){return t.addEventListener?(t.addEventListener(e,n,!0),{remove:function(){t.removeEventListener(e,n,!0)}}):{remove:r}},registerDefault:function(){}};t.exports=i},function(t,e,n){"use strict";function r(t){try{t.focus()}catch(t){}}t.exports=r},function(t,e,n){"use strict";function r(t){if(void 0===(t=t||("undefined"!=typeof document?document:void 0)))return null;try{return t.activeElement||t.body}catch(e){return t.body}}t.exports=r},function(t,e){function n(){throw new Error("setTimeout has not been defined")}function r(){throw new Error("clearTimeout has not been defined")}function i(t){if(l===setTimeout)return setTimeout(t,0);if((l===n||!l)&&setTimeout)return l=setTimeout,setTimeout(t,0);try{return l(t,0)}catch(e){try{return l.call(null,t,0)}catch(e){return l.call(this,t,0)}}}function o(t){if(f===clearTimeout)return clearTimeout(t);if((f===r||!f)&&clearTimeout)return f=clearTimeout,clearTimeout(t);try{return f(t)}catch(e){try{return f.call(null,t)}catch(e){return f.call(this,t)}}}function a(){v&&h&&(v=!1,h.length?d=h.concat(d):g=-1,d.length&&u())}function u(){if(!v){var t=i(a);v=!0;for(var e=d.length;e;){for(h=d,d=[];++g<e;)h&&h[g].run();g=-1,e=d.length}h=null,v=!1,o(t)}}function c(t,e){this.fun=t,this.array=e}function s(){}var l,f,p=t.exports={};!function(){try{l="function"==typeof setTimeout?setTimeout:n}catch(t){l=n}try{f="function"==typeof clearTimeout?clearTimeout:r}catch(t){f=r}}();var h,d=[],v=!1,g=-1;p.nextTick=function(t){var e=new Array(arguments.length-1);if(arguments.length>1)for(var n=1;n<arguments.length;n++)e[n-1]=arguments[n];d.push(new c(t,e)),1!==d.length||v||i(u)},c.prototype.run=function(){this.fun.apply(null,this.array)},p.title="browser",p.browser=!0,p.env={},p.argv=[],p.version="",p.versions={},p.on=s,p.addListener=s,p.once=s,p.off=s,p.removeListener=s,p.removeAllListeners=s,p.emit=s,p.prependListener=s,p.prependOnceListener=s,p.listeners=function(t){return[]},p.binding=function(t){throw new Error("process.binding is not supported")},p.cwd=function(){return"/"},p.chdir=function(t){throw new Error("process.chdir is not supported")},p.umask=function(){return 0}},function(t,e,n){"use strict";var r=n(343);t.exports=function(t){return r(t,!1)}},function(t,e,n){"use strict";function r(t,e){return t+e.charAt(0).toUpperCase()+e.substring(1)}var i={animationIterationCount:!0,borderImageOutset:!0,borderImageSlice:!0,borderImageWidth:!0,boxFlex:!0,boxFlexGroup:!0,boxOrdinalGroup:!0,columnCount:!0,columns:!0,flex:!0,flexGrow:!0,flexPositive:!0,flexShrink:!0,flexNegative:!0,flexOrder:!0,gridRow:!0,gridRowEnd:!0,gridRowSpan:!0,gridRowStart:!0,gridColumn:!0,gridColumnEnd:!0,gridColumnSpan:!0,gridColumnStart:!0,fontWeight:!0,lineClamp:!0,lineHeight:!0,opacity:!0,order:!0,orphans:!0,tabSize:!0,widows:!0,zIndex:!0,zoom:!0,fillOpacity:!0,floodOpacity:!0,stopOpacity:!0,strokeDasharray:!0,strokeDashoffset:!0,strokeMiterlimit:!0,strokeOpacity:!0,strokeWidth:!0},o=["Webkit","ms","Moz","O"];Object.keys(i).forEach(function(t){o.forEach(function(e){i[r(e,t)]=i[t]})});var a={background:{backgroundAttachment:!0,backgroundColor:!0,backgroundImage:!0,backgroundPositionX:!0,backgroundPositionY:!0,backgroundRepeat:!0},backgroundPosition:{backgroundPositionX:!0,backgroundPositionY:!0},border:{borderWidth:!0,borderStyle:!0,borderColor:!0},borderBottom:{borderBottomWidth:!0,borderBottomStyle:!0,borderBottomColor:!0},borderLeft:{borderLeftWidth:!0,borderLeftStyle:!0,borderLeftColor:!0},borderRight:{borderRightWidth:!0,borderRightStyle:!0,borderRightColor:!0},borderTop:{borderTopWidth:!0,borderTopStyle:!0,borderTopColor:!0},font:{fontStyle:!0,fontVariant:!0,fontWeight:!0,fontSize:!0,lineHeight:!0,fontFamily:!0},outline:{outlineWidth:!0,outlineStyle:!0,outlineColor:!0}},u={isUnitlessNumber:i,shorthandPropertyExpansions:a};t.exports=u},function(t,e,n){"use strict";function r(t,e){if(!(t instanceof e))throw new TypeError("Cannot call a class as a function")}var i=n(1),o=n(18),a=(n(0),function(){function t(e){r(this,t),this._callbacks=null,this._contexts=null,this._arg=e}return t.prototype.enqueue=function(t,e){this._callbacks=this._callbacks||[],this._callbacks.push(t),this._contexts=this._contexts||[],this._contexts.push(e)},t.prototype.notifyAll=function(){var t=this._callbacks,e=this._contexts,n=this._arg;if(t&&e){t.length!==e.length&&i("24"),this._callbacks=null,this._contexts=null;for(var r=0;r<t.length;r++)t[r].call(e[r],n);t.length=0,e.length=0}},t.prototype.checkpoint=function(){return this._callbacks?this._callbacks.length:0},t.prototype.rollback=function(t){this._callbacks&&this._contexts&&(this._callbacks.length=t,this._contexts.length=t)},t.prototype.reset=function(){this._callbacks=null,this._contexts=null},t.prototype.destructor=function(){this.reset()},t}());t.exports=o.addPoolingTo(a)},function(t,e,n){"use strict";function r(t){return!!s.hasOwnProperty(t)||!c.hasOwnProperty(t)&&(u.test(t)?(s[t]=!0,!0):(c[t]=!0,!1))}function i(t,e){return null==e||t.hasBooleanValue&&!e||t.hasNumericValue&&isNaN(e)||t.hasPositiveNumericValue&&e<1||t.hasOverloadedBooleanValue&&!1===e}var o=n(21),a=(n(4),n(9),n(407)),u=(n(2),new RegExp("^["+o.ATTRIBUTE_NAME_START_CHAR+"]["+o.ATTRIBUTE_NAME_CHAR+"]*$")),c={},s={},l={createMarkupForID:function(t){return o.ID_ATTRIBUTE_NAME+"="+a(t)},setAttributeForID:function(t,e){t.setAttribute(o.ID_ATTRIBUTE_NAME,e)},createMarkupForRoot:function(){return o.ROOT_ATTRIBUTE_NAME+'=""'},setAttributeForRoot:function(t){t.setAttribute(o.ROOT_ATTRIBUTE_NAME,"")},createMarkupForProperty:function(t,e){var n=o.properties.hasOwnProperty(t)?o.properties[t]:null;if(n){if(i(n,e))return"";var r=n.attributeName;return n.hasBooleanValue||n.hasOverloadedBooleanValue&&!0===e?r+'=""':r+"="+a(e)}return o.isCustomAttribute(t)?null==e?"":t+"="+a(e):null},createMarkupForCustomAttribute:function(t,e){return r(t)&&null!=e?t+"="+a(e):""},setValueForProperty:function(t,e,n){var r=o.properties.hasOwnProperty(e)?o.properties[e]:null;if(r){var a=r.mutationMethod;if(a)a(t,n);else{if(i(r,n))return void this.deleteValueForProperty(t,e);if(r.mustUseProperty)t[r.propertyName]=n;else{var u=r.attributeName,c=r.attributeNamespace;c?t.setAttributeNS(c,u,""+n):r.hasBooleanValue||r.hasOverloadedBooleanValue&&!0===n?t.setAttribute(u,""):t.setAttribute(u,""+n)}}}else if(o.isCustomAttribute(e))return void l.setValueForAttribute(t,e,n)},setValueForAttribute:function(t,e,n){if(r(e)){null==n?t.removeAttribute(e):t.setAttribute(e,""+n)}},deleteValueForAttribute:function(t,e){t.removeAttribute(e)},deleteValueForProperty:function(t,e){var n=o.properties.hasOwnProperty(e)?o.properties[e]:null;if(n){var r=n.mutationMethod;if(r)r(t,void 0);else if(n.mustUseProperty){var i=n.propertyName;n.hasBooleanValue?t[i]=!1:t[i]=""}else t.removeAttribute(n.attributeName)}else o.isCustomAttribute(e)&&t.removeAttribute(e)}};t.exports=l},function(t,e,n){"use strict";var r={hasCachedChildNodes:1};t.exports=r},function(t,e,n){"use strict";function r(){if(this._rootNodeID&&this._wrapperState.pendingUpdate){this._wrapperState.pendingUpdate=!1;var t=this._currentElement.props,e=u.getValue(t);null!=e&&i(this,Boolean(t.multiple),e)}}function i(t,e,n){var r,i,o=c.getNodeFromInstance(t).options;if(e){for(r={},i=0;i<n.length;i++)r[""+n[i]]=!0;for(i=0;i<o.length;i++){var a=r.hasOwnProperty(o[i].value);o[i].selected!==a&&(o[i].selected=a)}}else{for(r=""+n,i=0;i<o.length;i++)if(o[i].value===r)return void(o[i].selected=!0);o.length&&(o[0].selected=!0)}}function o(t){var e=this._currentElement.props,n=u.executeOnChange(e,t);return this._rootNodeID&&(this._wrapperState.pendingUpdate=!0),s.asap(r,this),n}var a=n(3),u=n(86),c=n(4),s=n(12),l=(n(2),!1),f={getHostProps:function(t,e){return a({},e,{onChange:t._wrapperState.onChange,value:void 0})},mountWrapper:function(t,e){var n=u.getValue(e);t._wrapperState={pendingUpdate:!1,initialValue:null!=n?n:e.defaultValue,listeners:null,onChange:o.bind(t),wasMultiple:Boolean(e.multiple)},void 0===e.value||void 0===e.defaultValue||l||(l=!0)},getSelectValueContext:function(t){return t._wrapperState.initialValue},postUpdateWrapper:function(t){var e=t._currentElement.props;t._wrapperState.initialValue=void 0;var n=t._wrapperState.wasMultiple;t._wrapperState.wasMultiple=Boolean(e.multiple);var r=u.getValue(e);null!=r?(t._wrapperState.pendingUpdate=!1,i(t,Boolean(e.multiple),r)):n!==Boolean(e.multiple)&&(null!=e.defaultValue?i(t,Boolean(e.multiple),e.defaultValue):i(t,Boolean(e.multiple),e.multiple?[]:""))}};t.exports=f},function(t,e,n){"use strict";var r,i={injectEmptyComponentFactory:function(t){r=t}},o={create:function(t){return r(t)}};o.injection=i,t.exports=o},function(t,e,n){"use strict";var r={logTopLevelRenders:!1};t.exports=r},function(t,e,n){"use strict";function r(t){return u||a("111",t.type),new u(t)}function i(t){return new c(t)}function o(t){return t instanceof c}var a=n(1),u=(n(0),null),c=null,s={injectGenericComponentClass:function(t){u=t},injectTextComponentClass:function(t){c=t}},l={createInternalComponent:r,createInstanceForText:i,isTextComponent:o,injection:s};t.exports=l},function(t,e,n){"use strict";function r(t){return o(document.documentElement,t)}var i=n(367),o=n(331),a=n(154),u=n(155),c={hasSelectionCapabilities:function(t){var e=t&&t.nodeName&&t.nodeName.toLowerCase();return e&&("input"===e&&"text"===t.type||"textarea"===e||"true"===t.contentEditable)},getSelectionInformation:function(){var t=u();return{focusedElem:t,selectionRange:c.hasSelectionCapabilities(t)?c.getSelection(t):null}},restoreSelection:function(t){var e=u(),n=t.focusedElem,i=t.selectionRange;e!==n&&r(n)&&(c.hasSelectionCapabilities(n)&&c.setSelection(n,i),a(n))},getSelection:function(t){var e;if("selectionStart"in t)e={start:t.selectionStart,end:t.selectionEnd};else if(document.selection&&t.nodeName&&"input"===t.nodeName.toLowerCase()){var n=document.selection.createRange();n.parentElement()===t&&(e={start:-n.moveStart("character",-t.value.length),end:-n.moveEnd("character",-t.value.length)})}else e=i.getOffsets(t);return e||{start:0,end:0}},setSelection:function(t,e){var n=e.start,r=e.end;if(void 0===r&&(r=n),"selectionStart"in t)t.selectionStart=n,t.selectionEnd=Math.min(r,t.value.length);else if(document.selection&&t.nodeName&&"input"===t.nodeName.toLowerCase()){var o=t.createTextRange();o.collapse(!0),o.moveStart("character",n),o.moveEnd("character",r-n),o.select()}else i.setOffsets(t,e)}};t.exports=c},function(t,e,n){"use strict";function r(t,e){for(var n=Math.min(t.length,e.length),r=0;r<n;r++)if(t.charAt(r)!==e.charAt(r))return r;return t.length===e.length?-1:n}function i(t){return t?t.nodeType===D?t.documentElement:t.firstChild:null}function o(t){return t.getAttribute&&t.getAttribute(P)||""}function a(t,e,n,r,i){var o;if(x.logTopLevelRenders){var a=t._currentElement.props.child,u=a.type;o="React mount: "+("string"==typeof u?u:u.displayName||u.name),console.time(o)}var c=k.mountComponent(t,n,null,_(t,e),i,0);o&&console.timeEnd(o),t._renderedComponent._topLevelWrapper=t,j._mountImageIntoNode(c,e,t,r,n)}function u(t,e,n,r){var i=M.ReactReconcileTransaction.getPooled(!n&&b.useCreateElement);i.perform(a,null,t,e,i,n,r),M.ReactReconcileTransaction.release(i)}function c(t,e,n){for(k.unmountComponent(t,n),e.nodeType===D&&(e=e.documentElement);e.lastChild;)e.removeChild(e.lastChild)}function s(t){var e=i(t);if(e){var n=y.getInstanceFromNode(e);return!(!n||!n._hostParent)}}function l(t){return!(!t||t.nodeType!==I&&t.nodeType!==D&&t.nodeType!==R)}function f(t){var e=i(t),n=e&&y.getInstanceFromNode(e);return n&&!n._hostParent?n:null}function p(t){var e=f(t);return e?e._hostContainerInfo._topLevelWrapper:null}var h=n(1),d=n(20),v=n(21),g=n(26),m=n(53),y=(n(15),n(4)),_=n(361),b=n(363),x=n(164),w=n(39),C=(n(9),n(377)),k=n(24),E=n(89),M=n(12),T=n(51),S=n(174),N=(n(0),n(57)),A=n(96),P=(n(2),v.ID_ATTRIBUTE_NAME),O=v.ROOT_ATTRIBUTE_NAME,I=1,D=9,R=11,L={},U=1,F=function(){this.rootID=U++};F.prototype.isReactComponent={},F.prototype.render=function(){return this.props.child},F.isReactTopLevelWrapper=!0;var j={TopLevelWrapper:F,_instancesByReactRootID:L,scrollMonitor:function(t,e){e()},_updateRootComponent:function(t,e,n,r,i){return j.scrollMonitor(r,function(){E.enqueueElementInternal(t,e,n),i&&E.enqueueCallbackInternal(t,i)}),t},_renderNewRootComponent:function(t,e,n,r){l(e)||h("37"),m.ensureScrollValueMonitoring();var i=S(t,!1);M.batchedUpdates(u,i,e,n,r);var o=i._instance.rootID;return L[o]=i,i},renderSubtreeIntoContainer:function(t,e,n,r){return null!=t&&w.has(t)||h("38"),j._renderSubtreeIntoContainer(t,e,n,r)},_renderSubtreeIntoContainer:function(t,e,n,r){E.validateCallback(r,"ReactDOM.render"),g.isValidElement(e)||h("39","string"==typeof e?" Instead of passing a string like 'div', pass React.createElement('div') or <div />.":"function"==typeof e?" Instead of passing a class like Foo, pass React.createElement(Foo) or <Foo />.":null!=e&&void 0!==e.props?" This may be caused by unintentionally loading two independent copies of React.":"");var a,u=g.createElement(F,{child:e});if(t){var c=w.get(t);a=c._processChildContext(c._context)}else a=T;var l=p(n);if(l){var f=l._currentElement,d=f.props.child;if(A(d,e)){var v=l._renderedComponent.getPublicInstance(),m=r&&function(){r.call(v)};return j._updateRootComponent(l,u,a,n,m),v}j.unmountComponentAtNode(n)}var y=i(n),_=y&&!!o(y),b=s(n),x=_&&!l&&!b,C=j._renderNewRootComponent(u,n,x,a)._renderedComponent.getPublicInstance();return r&&r.call(C),C},render:function(t,e,n){return j._renderSubtreeIntoContainer(null,t,e,n)},unmountComponentAtNode:function(t){l(t)||h("40");var e=p(t);if(!e){s(t),1===t.nodeType&&t.hasAttribute(O);return!1}return delete L[e._instance.rootID],M.batchedUpdates(c,e,t,!1),!0},_mountImageIntoNode:function(t,e,n,o,a){if(l(e)||h("41"),o){var u=i(e);if(C.canReuseMarkup(t,u))return void y.precacheNode(n,u);var c=u.getAttribute(C.CHECKSUM_ATTR_NAME);u.removeAttribute(C.CHECKSUM_ATTR_NAME);var s=u.outerHTML;u.setAttribute(C.CHECKSUM_ATTR_NAME,c);var f=t,p=r(f,s),v=" (client) "+f.substring(p-20,p+20)+"\n (server) "+s.substring(p-20,p+20);e.nodeType===D&&h("42",v)}if(e.nodeType===D&&h("43"),a.useCreateElement){for(;e.lastChild;)e.removeChild(e.lastChild);d.insertTreeBefore(e,t,null)}else N(e,t),y.precacheNode(n,e.firstChild)}};t.exports=j},function(t,e,n){"use strict";var r=n(1),i=n(26),o=(n(0),{HOST:0,COMPOSITE:1,EMPTY:2,getType:function(t){return null===t||!1===t?o.EMPTY:i.isValidElement(t)?"function"==typeof t.type?o.COMPOSITE:o.HOST:void r("26",t)}});t.exports=o},function(t,e,n){"use strict";function r(t,e){return null==e&&i("30"),null==t?e:Array.isArray(t)?Array.isArray(e)?(t.push.apply(t,e),t):(t.push(e),t):Array.isArray(e)?[t].concat(e):[t,e]}var i=n(1);n(0);t.exports=r},function(t,e,n){"use strict";function r(t,e,n){Array.isArray(t)?t.forEach(e,n):t&&e.call(n,t)}t.exports=r},function(t,e,n){"use strict";function r(t){for(var e;(e=t._renderedNodeType)===i.COMPOSITE;)t=t._renderedComponent;return e===i.HOST?t._renderedComponent:e===i.EMPTY?null:void 0}var i=n(168);t.exports=r},function(t,e,n){"use strict";function r(){return!o&&i.canUseDOM&&(o="textContent"in document.documentElement?"textContent":"innerText"),o}var i=n(6),o=null;t.exports=r},function(t,e,n){"use strict";function r(t){var e=t.type,n=t.nodeName;return n&&"input"===n.toLowerCase()&&("checkbox"===e||"radio"===e)}function i(t){return t._wrapperState.valueTracker}function o(t,e){t._wrapperState.valueTracker=e}function a(t){t._wrapperState.valueTracker=null}function u(t){var e;return t&&(e=r(t)?""+t.checked:t.value),e}var c=n(4),s={_getTrackerFromNode:function(t){return i(c.getInstanceFromNode(t))},track:function(t){if(!i(t)){var e=c.getNodeFromInstance(t),n=r(e)?"checked":"value",u=Object.getOwnPropertyDescriptor(e.constructor.prototype,n),s=""+e[n];e.hasOwnProperty(n)||"function"!=typeof u.get||"function"!=typeof u.set||(Object.defineProperty(e,n,{enumerable:u.enumerable,configurable:!0,get:function(){return u.get.call(this)},set:function(t){s=""+t,u.set.call(this,t)}}),o(t,{getValue:function(){return s},setValue:function(t){s=""+t},stopTracking:function(){a(t),delete e[n]}}))}},updateValueIfChanged:function(t){if(!t)return!1;var e=i(t);if(!e)return s.track(t),!0;var n=e.getValue(),r=u(c.getNodeFromInstance(t));return r!==n&&(e.setValue(r),!0)},stopTracking:function(t){var e=i(t);e&&e.stopTracking()}};t.exports=s},function(t,e,n){"use strict";function r(t){if(t){var e=t.getName();if(e)return" Check the render method of `"+e+"`."}return""}function i(t){return"function"==typeof t&&void 0!==t.prototype&&"function"==typeof t.prototype.mountComponent&&"function"==typeof t.prototype.receiveComponent}function o(t,e){var n;if(null===t||!1===t)n=s.create(o);else if("object"==typeof t){var u=t,c=u.type;if("function"!=typeof c&&"string"!=typeof c){var p="";p+=r(u._owner),a("130",null==c?c:typeof c,p)}"string"==typeof u.type?n=l.createInternalComponent(u):i(u.type)?(n=new u.type(u),n.getHostNode||(n.getHostNode=n.getNativeNode)):n=new f(u)}else"string"==typeof t||"number"==typeof t?n=l.createInstanceForText(t):a("131",typeof t);return n._mountIndex=0,n._mountImage=null,n}var a=n(1),u=n(3),c=n(358),s=n(163),l=n(165),f=(n(420),n(0),n(2),function(t){this.construct(t)});u(f.prototype,c,{_instantiateReactComponent:o}),t.exports=o},function(t,e,n){"use strict";function r(t){var e=t&&t.nodeName&&t.nodeName.toLowerCase();return"input"===e?!!i[t.type]:"textarea"===e}var i={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};t.exports=r},function(t,e,n){"use strict";var r=n(6),i=n(56),o=n(57),a=function(t,e){if(e){var n=t.firstChild;if(n&&n===t.lastChild&&3===n.nodeType)return void(n.nodeValue=e)}t.textContent=e};r.canUseDOM&&("textContent"in document.documentElement||(a=function(t,e){if(3===t.nodeType)return void(t.nodeValue=e);o(t,i(e))})),t.exports=a},function(t,e,n){"use strict";function r(t,e){return t&&"object"==typeof t&&null!=t.key?s.escape(t.key):e.toString(36)}function i(t,e,n,o){var p=typeof t;if("undefined"!==p&&"boolean"!==p||(t=null),null===t||"string"===p||"number"===p||"object"===p&&t.$$typeof===u)return n(o,t,""===e?l+r(t,0):e),1;var h,d,v=0,g=""===e?l:e+f;if(Array.isArray(t))for(var m=0;m<t.length;m++)h=t[m],d=g+r(h,m),v+=i(h,d,n,o);else{var y=c(t);if(y){var _,b=y.call(t);if(y!==t.entries)for(var x=0;!(_=b.next()).done;)h=_.value,d=g+r(h,x++),v+=i(h,d,n,o);else for(;!(_=b.next()).done;){var w=_.value;w&&(h=w[1],d=g+s.escape(w[0])+f+r(h,0),v+=i(h,d,n,o))}}else if("object"===p){var C="",k=String(t);a("31","[object Object]"===k?"object with keys {"+Object.keys(t).join(", ")+"}":k,C)}}return v}function o(t,e,n){return null==t?0:i(t,"",e,n)}var a=n(1),u=(n(15),n(373)),c=n(404),s=(n(0),n(85)),l=(n(2),"."),f=":";t.exports=o},function(t,e,n){"use strict";function r(t,e,n){this.props=t,this.context=e,this.refs=s,this.updater=n||c}function i(t,e,n){this.props=t,this.context=e,this.refs=s,this.updater=n||c}function o(){}var a=n(40),u=n(3),c=n(181),s=(n(182),n(51));n(0),n(421);r.prototype.isReactComponent={},r.prototype.setState=function(t,e){"object"!=typeof t&&"function"!=typeof t&&null!=t&&a("85"),this.updater.enqueueSetState(this,t),e&&this.updater.enqueueCallback(this,e,"setState")},r.prototype.forceUpdate=function(t){this.updater.enqueueForceUpdate(this),t&&this.updater.enqueueCallback(this,t,"forceUpdate")};o.prototype=r.prototype,i.prototype=new o,i.pro
