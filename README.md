```python
import pandas as pd
import numpy as np
import time
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

from collections import Counter
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, KBinsDiscretizer
from sklearn.metrics import *
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#classifiers
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

from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek

from sklearn.tree._tree import TREE_LEAF

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge, Lasso
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_val_score, cross_validate, StratifiedKFold, RandomizedSearchCV, GridSearchCV

from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

%matplotlib inline
warnings.filterwarnings('ignore')
```


```python
#Load Data
df = pd.read_csv('adult.csv')
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
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education.num</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital.gain</th>
      <th>capital.loss</th>
      <th>hours.per.week</th>
      <th>native.country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90</td>
      <td>?</td>
      <td>77053</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>?</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>82</td>
      <td>Private</td>
      <td>132870</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>18</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>66</td>
      <td>?</td>
      <td>186061</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Widowed</td>
      <td>?</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54</td>
      <td>Private</td>
      <td>140359</td>
      <td>7th-8th</td>
      <td>4</td>
      <td>Divorced</td>
      <td>Machine-op-inspct</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>3900</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>Private</td>
      <td>264663</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Separated</td>
      <td>Prof-specialty</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>3900</td>
      <td>40</td>
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
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education.num</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital.gain</th>
      <th>capital.loss</th>
      <th>hours.per.week</th>
      <th>native.country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32556</th>
      <td>22</td>
      <td>Private</td>
      <td>310152</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Never-married</td>
      <td>Protective-serv</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32557</th>
      <td>27</td>
      <td>Private</td>
      <td>257302</td>
      <td>Assoc-acdm</td>
      <td>12</td>
      <td>Married-civ-spouse</td>
      <td>Tech-support</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32558</th>
      <td>40</td>
      <td>Private</td>
      <td>154374</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>32559</th>
      <td>58</td>
      <td>Private</td>
      <td>151910</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Adm-clerical</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32560</th>
      <td>22</td>
      <td>Private</td>
      <td>201490</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
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
    workclass         32561 non-null object
    fnlwgt            32561 non-null int64
    education         32561 non-null object
    education.num     32561 non-null int64
    marital.status    32561 non-null object
    occupation        32561 non-null object
    relationship      32561 non-null object
    race              32561 non-null object
    sex               32561 non-null object
    capital.gain      32561 non-null int64
    capital.loss      32561 non-null int64
    hours.per.week    32561 non-null int64
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
    Trinadad&Tobago                  19
    Cambodia                         19
    Thailand                         18
    Laos                             18
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




![png](./output/output_5_1.png)



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




    <matplotlib.axes._subplots.AxesSubplot at 0x22595548860>




![png](./output/output_6_1.png)



```python
#Relation of Each Feature with the target
fig, (a,b)= plt.subplots(1,2,figsize=(20,6))
sns.boxplot(y='hours.per.week',x='income',data=df,ax=a)
sns.boxplot(y='age',x='income',data=df,ax=b)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2248a2015c0>




![png](./output/output_7_1.png)



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


    workclass
    ['?' 'Private' 'State-gov' 'Federal-gov' 'Self-emp-not-inc' 'Self-emp-inc'
     'Local-gov' 'Without-pay' 'Never-worked']


    fnlwgt
    [ 77053 132870 186061 ...  34066  84661 257302]


    education
    ['HS-grad' 'Some-college' '7th-8th' '10th' 'Doctorate' 'Prof-school'
     'Bachelors' 'Masters' '11th' 'Assoc-acdm' 'Assoc-voc' '1st-4th' '5th-6th'
     '12th' '9th' 'Preschool']


    education.num
    [ 9 10  4  6 16 15 13 14  7 12 11  2  3  8  5  1]


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
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education.num</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital.gain</th>
      <th>capital.loss</th>
      <th>hours.per.week</th>
      <th>native.country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>32561.000000</td>
      <td>32561</td>
      <td>3.256100e+04</td>
      <td>32561</td>
      <td>32561.000000</td>
      <td>32561</td>
      <td>32561</td>
      <td>32561</td>
      <td>32561</td>
      <td>32561</td>
      <td>32561.000000</td>
      <td>32561.000000</td>
      <td>32561.000000</td>
      <td>32561</td>
      <td>32561</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>9</td>
      <td>NaN</td>
      <td>16</td>
      <td>NaN</td>
      <td>7</td>
      <td>15</td>
      <td>6</td>
      <td>5</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>42</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>Private</td>
      <td>NaN</td>
      <td>HS-grad</td>
      <td>NaN</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>22696</td>
      <td>NaN</td>
      <td>10501</td>
      <td>NaN</td>
      <td>14976</td>
      <td>4140</td>
      <td>13193</td>
      <td>27816</td>
      <td>21790</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29170</td>
      <td>24720</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38.581647</td>
      <td>NaN</td>
      <td>1.897784e+05</td>
      <td>NaN</td>
      <td>10.080679</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1077.648844</td>
      <td>87.303830</td>
      <td>40.437456</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.640433</td>
      <td>NaN</td>
      <td>1.055500e+05</td>
      <td>NaN</td>
      <td>2.572720</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7385.292085</td>
      <td>402.960219</td>
      <td>12.347429</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.000000</td>
      <td>NaN</td>
      <td>1.228500e+04</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>28.000000</td>
      <td>NaN</td>
      <td>1.178270e+05</td>
      <td>NaN</td>
      <td>9.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>37.000000</td>
      <td>NaN</td>
      <td>1.783560e+05</td>
      <td>NaN</td>
      <td>10.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48.000000</td>
      <td>NaN</td>
      <td>2.370510e+05</td>
      <td>NaN</td>
      <td>12.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>45.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>90.000000</td>
      <td>NaN</td>
      <td>1.484705e+06</td>
      <td>NaN</td>
      <td>16.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>99999.000000</td>
      <td>4356.000000</td>
      <td>99.000000</td>
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
    #T_54310ca6_4196_11ea_b31a_b88687537bddrow0_col0 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow0_col1 {
            color:  red;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow0_col2 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow0_col3 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow0_col4 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow1_col0 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow1_col1 {
            color:  red;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow1_col2 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow1_col3 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow1_col4 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow2_col0 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow2_col1 {
            color:  red;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow2_col2 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow2_col3 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow2_col4 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow3_col0 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow3_col1 {
            color:  red;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow3_col2 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow3_col3 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow3_col4 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow4_col0 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow4_col1 {
            color:  red;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow4_col2 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow4_col3 {
            color:  black;
        }    #T_54310ca6_4196_11ea_b31a_b88687537bddrow4_col4 {
            color:  black;
        }</style><table id="T_54310ca6_4196_11ea_b31a_b88687537bdd" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >age</th>        <th class="col_heading level0 col1" >capital.gain</th>        <th class="col_heading level0 col2" >capital.loss</th>        <th class="col_heading level0 col3" >education.num</th>        <th class="col_heading level0 col4" >fnlwgt</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_54310ca6_4196_11ea_b31a_b88687537bddlevel0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow0_col0" class="data row0 col0" >90</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow0_col1" class="data row0 col1" >0</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow0_col2" class="data row0 col2" >4356</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow0_col3" class="data row0 col3" >9</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow0_col4" class="data row0 col4" >77053</td>
            </tr>
            <tr>
                        <th id="T_54310ca6_4196_11ea_b31a_b88687537bddlevel0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow1_col0" class="data row1 col0" >82</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow1_col1" class="data row1 col1" >0</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow1_col2" class="data row1 col2" >4356</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow1_col3" class="data row1 col3" >9</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow1_col4" class="data row1 col4" >132870</td>
            </tr>
            <tr>
                        <th id="T_54310ca6_4196_11ea_b31a_b88687537bddlevel0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow2_col0" class="data row2 col0" >66</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow2_col1" class="data row2 col1" >0</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow2_col2" class="data row2 col2" >4356</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow2_col3" class="data row2 col3" >10</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow2_col4" class="data row2 col4" >186061</td>
            </tr>
            <tr>
                        <th id="T_54310ca6_4196_11ea_b31a_b88687537bddlevel0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow3_col0" class="data row3 col0" >54</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow3_col1" class="data row3 col1" >0</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow3_col2" class="data row3 col2" >3900</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow3_col3" class="data row3 col3" >4</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow3_col4" class="data row3 col4" >140359</td>
            </tr>
            <tr>
                        <th id="T_54310ca6_4196_11ea_b31a_b88687537bddlevel0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow4_col0" class="data row4 col0" >41</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow4_col1" class="data row4 col1" >0</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow4_col2" class="data row4 col2" >3900</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow4_col3" class="data row4 col3" >10</td>
                        <td id="T_54310ca6_4196_11ea_b31a_b88687537bddrow4_col4" class="data row4 col4" >264663</td>
            </tr>
    </tbody></table>


## Missing Values



```python
missingno.matrix(df, figsize=(12, 5))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22484e019e8>




![png](./output/output_15_1.png)



```python
df.isnull().sum()
```




    age               0
    workclass         0
    fnlwgt            0
    education         0
    education.num     0
    marital.status    0
    occupation        0
    relationship      0
    race              0
    sex               0
    capital.gain      0
    capital.loss      0
    hours.per.week    0
    native.country    0
    income            0
    dtype: int64




```python
df = pd.read_csv('adult.csv')

#convert '?' to NaN
df[df == '?'] = np.nan

#fill the NaN values with the mode. It could be filled with median, mean, etc
for col in df.columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

display(df.head())    

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
# The Iterative Imputer is developed by Scikit-Learn and models each feature with missing values as a function of other features. It uses that as an estimate for imputation. At each step, a feature is selected as ./output/output y and all other features are treated as inputs X. A regressor is then fitted on X and y and used to predict the missing values of y. This is done for each feature and repeated for several imputation rounds.
# The great thing about this method is that it allows you to use an estimator of your choosing. I used a RandomForestRegressor to mimic the behavior of the frequently used missForest in R.
imp = IterativeImputer(RandomForestRegressor(), max_iter=10, random_state=0)
df = pd.DataFrame(imp.fit_transform(df), columns=df.columns)

#If you have sufficient data, then it might be an attractive option to simply delete samples with missing data. However, keep in mind that it could create bias in your data. Perhaps the missing data follows a pattern that you miss out on.

#The Iterative Imputer allows for different estimators to be used. After some testing, I found out that you can even use Catboost as an estimator! Unfortunately, LightGBM and XGBoost do not work since their random state names differ.
"""
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
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education.num</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital.gain</th>
      <th>capital.loss</th>
      <th>hours.per.week</th>
      <th>native.country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90</td>
      <td>Private</td>
      <td>77053</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Prof-specialty</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>82</td>
      <td>Private</td>
      <td>132870</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>18</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>66</td>
      <td>Private</td>
      <td>186061</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Widowed</td>
      <td>Prof-specialty</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54</td>
      <td>Private</td>
      <td>140359</td>
      <td>7th-8th</td>
      <td>4</td>
      <td>Divorced</td>
      <td>Machine-op-inspct</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>3900</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>Private</td>
      <td>264663</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Separated</td>
      <td>Prof-specialty</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>3900</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>





    '\n# It is also possible to run a bivariate imputer (iterative imputer). However, it is needed to do labelencoding first. The code below enables us to run the imputer with a Random Forest estimator\n# The Iterative Imputer is developed by Scikit-Learn and models each feature with missing values as a function of other features. It uses that as an estimate for imputation. At each step, a feature is selected as ./output/output y and all other features are treated as inputs X. A regressor is then fitted on X and y and used to predict the missing values of y. This is done for each feature and repeated for several imputation rounds.\n# The great thing about this method is that it allows you to use an estimator of your choosing. I used a RandomForestRegressor to mimic the behavior of the frequently used missForest in R.\nimp = IterativeImputer(RandomForestRegressor(), max_iter=10, random_state=0)\ndf = pd.DataFrame(imp.fit_transform(df), columns=df.columns)\n\n#If you have sufficient data, then it might be an attractive option to simply delete samples with missing data. However, keep in mind that it could create bias in your data. Perhaps the missing data follows a pattern that you miss out on.\n\n#The Iterative Imputer allows for different estimators to be used. After some testing, I found out that you can even use Catboost as an estimator! Unfortunately, LightGBM and XGBoost do not work since their random state names differ.\n'



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

### Binning
Binning continuous variables prevents overfitting which is a common problem for tree based models like decision trees and random forest. It also helps dealing with outliers.

Let's binning the age and see later if the results improved or not.

Binning with fixed-width intervals (good for uniform distributions). For skewed distributions (not normal), we can use binning adaptive (quantile based).


```python
#Binner adaptive
binner = KBinsDiscretizer(encode='ordinal')
binner.fit(df[['age']])
df['age'] = binner.transform(df[['age']])

"""
#Using fixed-width intervals
age_labels = ['infant','child','teenager','young_adult','adult','old', 'very_old']

ranges = [0,5,12,18,35,60,81,100]

df['age'] = pd.cut(df.age, ranges, labels = age_labels)

"""
```




    "\n#Using fixed-width intervals\nage_labels = ['infant','child','teenager','young_adult','adult','old', 'very_old']\n\nranges = [0,5,12,18,35,60,81,100]\n\ndf['age'] = pd.cut(df.age, ranges, labels = age_labels)\n\n"



### Label Encoding


```python
categorical = ['age', 'workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
numerical = ['fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
```


```python
#using labelencoder
le = LabelEncoder()
for feature in categorical:
        df[feature] = le.fit_transform(df[feature])
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


![png](./output/output_29_0.png)



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

    2456 outliers were eliminated



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
    <=50K    22874
    >50K      7207
    dtype: int64




![png](./output/output_33_1.png)



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




    <matplotlib.axes._subplots.AxesSubplot at 0x2253f6fa6d8>




![png](./output/output_38_1.png)


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


![png](./output/output_41_0.png)



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
      <td>8.802656</td>
    </tr>
    <tr>
      <th>capital.loss</th>
      <td>4.065863</td>
    </tr>
    <tr>
      <th>fnlwgt</th>
      <td>1.260418</td>
    </tr>
    <tr>
      <th>hours.per.week</th>
      <td>0.101902</td>
    </tr>
    <tr>
      <th>education.num</th>
      <td>-0.201827</td>
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


![png](./output/output_43_0.png)



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
      <td>3.808393</td>
    </tr>
    <tr>
      <th>capital.gain</th>
      <td>2.356419</td>
    </tr>
    <tr>
      <th>hours.per.week</th>
      <td>0.101902</td>
    </tr>
    <tr>
      <th>education.num</th>
      <td>-0.201827</td>
    </tr>
    <tr>
      <th>fnlwgt</th>
      <td>-0.904787</td>
    </tr>
  </tbody>
</table>
</div>


# Feature Selection
- Feature Selection/Reduction (Remove useless features) -> See feature importance, correlations, Dimensionality reduction,

As we only have 14 features, we are not pressured to make a feature selection/reduction in order to increase drastically the computing time of the algorithms. So, for now, we are going to investigate if there are features extremely correlated to each other. After tuning and choosing the best model, we are revisiting feature selection methods just in case we face overfitting or to see if we could achieve the same results with the chosen model but with fewer features.

## Correlation

#Simple regression line of various variables in relation to  one other
sns.pairplot(X, x_vars = ['capital.loss', 'hours.per.week', 'education.num'], y_vars = 'age', size = 7, aspect = 0.7, kind = 'reg')

# Correlation between variables each other. It may take a while, not advised if there are plenty of features
sns.pairplot(data = X)


```python
plt.subplots(figsize=(20,7))
sns.heatmap(X.corr(method = 'pearson'),annot=True,cmap='coolwarm') # the method can also be 'spearman' or kendall'

#to see the correlation between just two variables
#df['age'].corr(df['capital.gain'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x224ee0b87b8>




![png](./output/output_48_1.png)


**Generally all of the features are not highly correlated (no correlations above abs(0.5)). The strongest correlations here is "sex"vs"relationship". Down bellow are the procedures to drop the highest correlated features, even though it's not needed in this case or it might even result in worse results. Let's proceed with PCA and Backward Elimination methods to really see if it is needed or not**

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
    0.15216510543494924
    0.09015381833712588
    0.08136512212145743
    0.07897212981640246
    0.07528183598157549
    0.07170978155854005
    0.0708024996157458
    0.06869831846254966
    0.06429338053471736
    0.06378453810296272
    0.05974537699557438
    0.049697465660073936
    0.04600209467325965
    0.027328532705065776


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


![png](./output/output_53_0.png)



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

    A total of 13 features selected: ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week']


Backward elimination actually tells us that we can eliminate the feature native.country. However, as we have just a few features, we don't have the urge to sacrifice a little bit of the accuracy of the model due to performing speed. In the latter parts of this notebook there are shown more advanced feature selection techniques just in case will be needed in the future.

# Base Line Model


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
    0  20317  0
    1  20317  0
    Accuracy: 0.5





    '\n#If we just want to perform OLS regression to find out interesting stats:\nX1 = sm.add_constant(X)\nmodel = sm.OLS(Y, X1).fit()\ndisplay(model.summary())\nprint(model.pvalues)\n'



# One Hot Encoding


```python
encoded_feat = pd.DataFrame(OneHotEncoder(categories="auto").fit_transform(df[categorical]).toarray())

#concatenate without adding null values -.-''
l1=encoded_feat.values.tolist()
l2=df[numerical].values.tolist()
for i in range(len(l1)):
    l1[i].extend(l2[i])

X=pd.DataFrame(l1,columns=encoded_feat.columns.tolist()+df[numerical].columns.tolist())

```

# Scaling Normalization

StandardScaler removes the mean and scales the data to unit variance. However, the outliers have an influence when computing the empirical mean and standard deviation which shrink the range of the feature values as shown in the left figure below. Note in particular that because the outliers on each feature have different magnitudes, the spread of the transformed data on each feature is very different: most of the data lie in the [-2, 4] range for the transformed median income feature while the same data is squeezed in the smaller [-0.2, 0.2] range for the transformed number of households.

StandardScaler therefore cannot guarantee balanced feature scales in the presence of outliers.

MinMaxScaler rescales the data set such that all feature values are in the range [0, 1] as shown in the right panel below. However, this scaling compress all inliers in the narrow range [0, 0.005] for the transformed number of households.

[Source](https://stackoverflow.com/questions/51237635/difference-between-standard-scaler-and-minmaxscaler)


```python
#Standard
col_names = X.columns

features = X[col_names]

scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)

X[col_names] = features
```


```python
#you can try this one to see if it results in better performances
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

train,test = train_test_split(new_df,test_size=testSize)

y_train = train['income']
X_train = train.drop('income',axis=1)
y_test = test['income']
X_test = test.drop('income',axis=1)

models = []
classifiers = []
model = GaussianNB().fit(X_train,y_train)
models.append(('Naive Bayes', model))
classifiers.append(model)
```

# Building Classifiers and Finding their Best Parameters
Just tuning the crutial parameters, not all of them.
I intercalate between randomizedSearch and GridSearch to diversify


```python
tuning_num_folds = 10
jobs=4
num_random_state=10
scoring_criteria='accuracy'
```

## Random Forest Tuning


```python
params ={
             'max_depth': st.randint(3, 11),
            'n_estimators': [50,100,150,200,250],
             'max_features':["sqrt", "log2"],
             'max_leaf_nodes':st.randint(6, 10)
            }

skf = StratifiedKFold(n_splits=tuning_num_folds, shuffle = True)

random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=params, scoring='roc_auc', n_jobs=jobs, cv=skf.split(X_train,y_train), verbose=3, random_state=num_random_state)
random_search.fit(X_train,y_train)

tunned_model = RandomForestClassifier(n_estimators=random_search.best_params_['n_estimators'], max_features=random_search.best_params_['max_features'], max_leaf_nodes=random_search.best_params_['max_leaf_nodes'], max_depth=random_search.best_params_['max_depth'])

models.append(('Random Forest', tunned_model))
classifiers.append(tunned_model)

#save model
pickle.dump(tunned_model, open('./models/Random Forest.sav', 'wb'))
```

    Fitting 2 folds for each of 10 candidates, totalling 20 fits


    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done  20 out of  20 | elapsed:    4.4s remaining:    0.0s
    [Parallel(n_jobs=4)]: Done  20 out of  20 | elapsed:    4.4s finished


## Tuning Logistic Regression


```python
C_vals = [0.0001, 0.001, 0.01, 0.1,0.13,0.2, .15, .25, .275, .33, 0.5, .66, 0.75, 1.0, 2.5, 4.0,4.5,5.0,5.1,5.5,6.0, 10.0, 100.0, 1000.0]
penalties = ['l1','l2']
param = {'penalty': penalties, 'C': C_vals}

grid = GridSearchCV(LogisticRegression(), param,verbose=False, cv = StratifiedKFold(n_splits=tuning_num_folds,random_state=num_random_state,shuffle=True), n_jobs=jobs,scoring=scoring_criteria)
grid.fit(X_train,y_train)

tunned_model = LogisticRegression(penalty=grid.best_params_['penalty'], C=grid.best_params_['C'])
models.append(('Logistic Regression', tunned_model))
classifiers.append(tunned_model)

#save model
pickle.dump(tunned_model, open('./models/Logistic Regression.sav', 'wb'))
```

## MLPClassifier tuning



```python
param ={'max_iter': np.logspace(1, 5, 10).astype("int32"),
             'hidden_layer_sizes': np.logspace(2, 3, 4).astype("int32"),
             'activation':['identity', 'logistic', 'tanh', 'relu'],
             'learning_rate': ['adaptive'],
             'early_stopping': [True],
             'alpha': np.logspace(2, 3, 4).astype("int32")
            }

grid = GridSearchCV(MLPClassifier(), param,verbose=False, cv = StratifiedKFold(n_splits=tuning_num_folds,random_state=num_random_state,shuffle=True), n_jobs=jobs,scoring=scoring_criteria)
grid.fit(X_train,y_train)

models.append(('MLP', MLPClassifier(max_iter=grid.best_params_['max_iter'], hidden_layer_sizes=grid.best_params_['hidden_layer_sizes'], activation=grid.best_params_['activation'], early_stopping=grid.best_params_['early_stopping'], learning_rate=grid.best_params_['learning_rate'], alpha=grid.best_params_['alpha'])))
classifiers.append(MLPClassifier(max_iter=grid.best_params_['max_iter'], hidden_layer_sizes=grid.best_params_['hidden_layer_sizes'], activation=grid.best_params_['activation'], early_stopping=grid.best_params_['early_stopping'], learning_rate=grid.best_params_['learning_rate'], alpha=grid.best_params_['alpha']))
```

** KNeighborsClassifier tuning **

params ={'n_neighbors': st.randint(1,50),
            'weights':['uniform','distance']
            }

skf = StratifiedKFold(n_splits=tuning_num_folds, shuffle = True)

random_search = RandomizedSearchCV(KNeighborsClassifier(), param_distributions=params, scoring='roc_auc', n_jobs=jobs, cv=skf.split(X_train,y_train), verbose=3, random_state=num_random_state)
random_search.fit(X_train,y_train)

models.append(('KNN', KNeighborsClassifier(n_neighbors=random_search.best_params_['n_neighbors'], weights=random_search.best_params_['weights'])))
classifiers.append(KNeighborsClassifier(n_neighbors=random_search.best_params_['n_neighbors'], weights=random_search.best_params_['weights']))

** Linear Discriminant tuning **

param={'solver': ['svd', 'lsqr', 'eigen']}

grid = GridSearchCV(LinearDiscriminantAnalysis(), param,verbose=True, cv = StratifiedKFold(n_splits=tuning_num_folds,random_state=num_random_state,shuffle=True), n_jobs=jobs,scoring=scoring_criteria)
grid.fit(X_train,y_train)

models.append(('LDA', LinearDiscriminantAnalysis(solver=grid.best_params_['solver'])))
classifiers.append(LinearDiscriminantAnalysis(solver=grid.best_params_['solver']))

** DecisionTreeClassifier tuning **

Introduction to Feature Importance Graphic¶
Since each split in the decision tree distinguishes the dependent variable, splits closer to the root, aka starting point, have optimally been determined to have the greatest splitting effect. The feature importance graphic measures how much splitting impact each feature has. It is important to note that this by no means points to causality, but just like in hierarchical clustering, does point to a nebulous groups. Furthermore, for ensemble tree methods, feature impact is aggregated over all the trees.


```python
# Helper Function to visualize feature importance
plt.rcParams['figure.figsize'] = (8, 4)
predictors = [x for x in X.columns if x not in ['Survived']]
def feature_imp(DecisionTreeClassifier()):
    MO = model.fit(X_train, y_train)
    feat_imp = pd.Series(MO.feature_importances_, predictors).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
```

param = {'min_samples_split': [4,7,10,12], 'n_estimators': n_tree_range, 'n_estimators': [50,100,150,200,250]}

grid = GridSearchCV(DecisionTreeClassifier(), param,verbose=True, cv = StratifiedKFold(n_splits=tuning_num_folds,random_state=num_random_state,shuffle=True), n_jobs=jobs,scoring=scoring_criteria)
grid.fit(X_train,y_train)

models.append(('DTC', DecisionTreeClassifier(min_samples_split=grid.best_params_['min_samples_split'], n_estimators=grid.best_params_['n_estimators'])))
classifiers.append(DecisionTreeClassifier(min_samples_split=grid.best_params_['min_samples_split'], n_estimators=grid.best_params_['n_estimators']))


```python
**Pruning the decision trees**
Adapted from [here](https://stackoverflow.com/questions/49428469/pruning-decision-trees)
```


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

print(sum(clf.tree_.children_left < 0))
# start pruning from the root
prune_index(clf.tree_, 0, 5)
sum(clf.tree_.children_left < 0)

models.append(('DT', clf))
classifiers.append(clf)
```

** SVC tuning **

param={'C':[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5],
      'kernel': ["linear","rbf"]
      }

grid = GridSearchCV(SVC(), param,verbose=True, cv = StratifiedKFold(n_splits=tuning_num_folds,random_state=num_random_state,shuffle=True), n_jobs=jobs,scoring=scoring_criteria)
grid.fit(X_train,y_train)

models.append(('SVC', SVC(C=grid.best_params_['C'], kernel=grid.best_params_['kernel'])))
classifiers.append(SVC(C=grid.best_params_['C'], kernel=grid.best_params_['kernel']))

** SGDClassifier tuning **

param ={'loss':["hinge","log","modified_huber","epsilon_insensitive","squared_epsilon_insensitive"]}

grid = GridSearchCV(SGDClassifier(), param,verbose=False, cv = StratifiedKFold(n_splits=tuning_num_folds,random_state=num_random_state,shuffle=True), n_jobs=jobs,scoring=scoring_criteria)
grid.fit(X_train,y_train)

models.append(('SGD', SGDClassifier(loss=grid.best_params_['loss'])))
classifiers.append(SGDClassifier(loss=grid.best_params_['loss']))

## Boosting

### GradientBoostingClassifier tuning

param_grid ={'n_estimators': st.randint(100, 800),
            'loss': ['deviance', 'exponential'],
            'learning_rate': [0.1, 0.01,0.05,0.001],
            'max_depth': np.arange(2, 12, 1)}


random_search = RandomizedSearchCV(GradientBoostingClassifier(), param_distributions=params, scoring='roc_auc', n_jobs=jobs, cv=skf.split(X_train,y_train), verbose=3, random_state=num_random_state)
random_search.fit(X_train,y_train)

models.append(('GraBoost', GradientBoostingClassifier(n_estimators=random_search.best_params_['n_estimators'], loss=random_search.best_params_['loss'], max_depth=random_search.best_params_['max_depth'], learning_rate=random_search.best_params_['learning_rate'], verbose=0)))
classifiers.append(GradientBoostingClassifier(n_estimators=random_search.best_params_['n_estimators'], loss=random_search.best_params_['loss'], max_depth=random_search.best_params_['max_depth'], learning_rate=random_search.best_params_['learning_rate'], verbose=0))

### AdaBoostClassifier tuning

params ={'n_estimators':st.randint(100, 800),
        'learning_rate':np.arange(.1, 4, .5)}


random_search = RandomizedSearchCV(AdaBoostClassifier(), param_distributions=params, scoring='roc_auc', n_jobs=jobs, cv=skf.split(X_train,y_train), verbose=3, random_state=num_random_state)
random_search.fit(X_train,y_train)

models.append(('Ada', AdaBoostClassifier(n_estimators=random_search.best_params_['n_estimators'], learning_rate=random_search.best_params_['learning_rate'], verbose=0)))
classifiers.append(AdaBoostClassifier(n_estimators=random_search.best_params_['n_estimators'], learning_rate=random_search.best_params_['learning_rate'], verbose=0))

### CatBoostClassifier Tuning

param = {'iterations': [100, 150], 'learning_rate': [0.3, 0.4, 0.5]}

grid = GridSearchCV(CatBoostClassifier(), param,verbose=True, cv = StratifiedKFold(n_splits=tuning_num_folds,random_state=num_random_state,shuffle=True), n_jobs=jobs,scoring=scoring_criteria)
grid.fit(X_train,y_train)

models.append(('CAT', CatBoostClassifier(iterations=grid.best_params_['iterations'], learning_rate=grid.best_params_['learning_rate'], verbose=0)))
classifiers.append(CatBoostClassifier(iterations=grid.best_params_['iterations'], learning_rate=grid.best_params_['learning_rate'], verbose=0))

### XGBClassifier Tuning

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

skf = StratifiedKFold(n_splits=tuning_num_folds, shuffle = True)

random_search = RandomizedSearchCV(XGBClassifier(), param_distributions=params, scoring='roc_auc', n_jobs=jobs, cv=skf.split(X_train,y_train), verbose=3, random_state=num_random_state)
random_search.fit(X_train,y_train,early_stopping_rounds=50) #early stoppage for when there's no improvement

models.append(('XGB', XGBClassifier(colsample_bytree=random_search.best_params_['colsample_bytree'], gamma=random_search.best_params_['gamma'], max_depth=random_search.best_params_['max_depth'], min_child_weight=random_search.best_params_['min_child_weight'], subsample=random_search.best_params_['subsample'], verbose=0)))
classifiers.append(XGBClassifier(colsample_bytree=random_search.best_params_['colsample_bytree'], gamma=random_search.best_params_['gamma'], max_depth=random_search.best_params_['max_depth'], min_child_weight=random_search.best_params_['min_child_weight'], subsample=random_search.best_params_['subsample'], verbose=0))

# Finding the best Model


```python
def classify_performance(name, model, X_train, X_test, Y_train, Y_test, num_folds, scoring, seed, results, fprs, tprs, labels):
    stratifiedfold = StratifiedShuffleSplit(n_splits=num_folds, random_state=seed)
    cv_results = cross_validate(model, X_train, Y_train, cv=stratifiedfold, scoring=scoring, verbose=0)

    y_pred = model.predict(X_test)
    print("%s - Train: Accuracy mean - %.3f, Precision: %.3f, Recall: %.3f." % (name, cv_results['test_accuracy'].mean(), cv_results['test_precision'].mean(), cv_results['test_recall'].mean()))
    precision, recall, fscore, support = precision_recall_fscore_support(Y_test, y_pred,average='binary')
    print("%s - Test: Accuracy - %.3f, Precision: %.3f, Recall: %.3f, F-Score: %.3f" % (name, accuracy_score(Y_test, y_pred), precision,recall, fscore))

    skplt.metrics.plot_confusion_matrix(Y_test, y_pred, title="{} Confusion Matrix".format(name))
    plt.show()

    fpr, tpr, _ = roc_curve(Y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='{} ROC curve (area = {:.2})'.format(name, roc_auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.show()


    print('ROC AUC: %0.2f' % roc_auc)
    print("----------------------------------------------------------------------------------")

    fprs.append(fpr)
    tprs.append(tpr)
    labels.append('{} ROC curve (area = {:.2})'.format(name, roc_auc))
    new_entry = {'Model': name,'Test Accuracy': cv_results['test_accuracy'].mean(), 'Precision': precision, 'Recall': recall, 'F-Score': fscore, 'AUC': roc_auc,'Train Accuracy':cv_results['test_accuracy'].mean(), 'Train Precision': cv_results['test_precision'].mean(), 'Train Recall': cv_results['test_recall'].mean()}

    results = results.append(new_entry, ignore_index=True)

    return results, fprs, tprs, labels

```


```python
seed = 7
num_folds = 2
scoring = {'accuracy': 'accuracy',
           'precision': 'precision_macro',
           'recall': 'recall_macro'}

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, stratify=Y,random_state=seed)

results = pd.DataFrame(columns=['Model','Test Accuracy','Precision','Recall','F-Score', 'AUC','Train Accuracy'])

fprs = []
tprs = []
labels = []

for name, model in models:
    results, fprs, tprs, labels = classify_performance(name, model, X_train, X_test, Y_train, Y_test, num_folds, scoring, seed, results, fprs, tprs, labels)

```

    NB - Train: Accuracy mean - 0.808, Precision: 0.818, Recall: 0.808.
    NB - Test: Accuracy - 0.827, Precision: 0.832, Recall: 0.821, F-Score: 0.826



![png](./output/output_102_1.png)



![png](./output/output_102_2.png)


    ROC AUC: 0.83
    ----------------------------------------------------------------------------------
    logistic - Train: Accuracy mean - 0.859, Precision: 0.859, Recall: 0.859.
    logistic - Test: Accuracy - 0.861, Precision: 0.856, Recall: 0.867, F-Score: 0.862



![png](./output/output_102_4.png)



![png](./output/output_102_5.png)


    ROC AUC: 0.86
    ----------------------------------------------------------------------------------
    DT - Train: Accuracy mean - 0.807, Precision: 0.808, Recall: 0.807.
    DT - Test: Accuracy - 0.820, Precision: 0.824, Recall: 0.816, F-Score: 0.820



![png](./output/output_102_7.png)



![png](./output/output_102_8.png)


    ROC AUC: 0.82
    ----------------------------------------------------------------------------------
    XGBOOST - Train: Accuracy mean - 0.883, Precision: 0.883, Recall: 0.883.
    XGBOOST - Test: Accuracy - 0.883, Precision: 0.890, Recall: 0.874, F-Score: 0.882



![png](./output/output_102_10.png)



![png](./output/output_102_11.png)


    ROC AUC: 0.88
    ----------------------------------------------------------------------------------
    CATBOOST - Train: Accuracy mean - 0.889, Precision: 0.890, Recall: 0.889.
    CATBOOST - Test: Accuracy - 0.892, Precision: 0.900, Recall: 0.881, F-Score: 0.891



![png](./output/output_102_13.png)



![png](./output/output_102_14.png)


    ROC AUC: 0.89
    ----------------------------------------------------------------------------------


### Ensemble


```python

```


```python
copy_1 = results[results.Model != 'CAT']
copy_2 = results
copy_1.sort_values(by=["Test Accuracy"], ascending=False, inplace=True)
copy_2.sort_values(by=["Test Accuracy"], ascending=False, inplace=True)
```

Correlation among Base Models Predictions: How base models' predictions are correlated? If base models' predictions are weakly correlated with each other, the ensemble will likely to perform better. On the other hand, for a strong correlation of predictions among the base models, the ensemble will unlikely to perform better. To sumarize, diversity of predictions among the base models is inversely proportional to the ensemble accuracy. Let's make prediction for the test set.


```python
'''Create a data frame to store base models prediction.
First four in the dataframe are tree based models. Then two are kernel based. And the last is a linear model.'''
base_prediction = model_prediction # We've a df of all the models prediction.

"""Let's see how each model classifies a prticular class."""
bold('**All the Base Models Prediction:**')
display(base_prediction.head())

"""Let's visualize the correlations among the predictions of base models."""
plt.figure(figsize = (20, 6))
sns.heatmap(base_prediction.corr(), annot = True)
plt.title('Prediction Correlation among the Base Models', fontsize = 18)
plt.show()
```

Findings: The prediction looks quite similar for the 8 classifiers except when DT is compared to the others classifiers. Now we will create an ensemble with the base models RF, GBC, DT, KNN and LR. This ensemble can be called heterogeneous ensemble since we have three tree based, one kernel based and one linear models. We would use


```python

```

### Voting

Voting can be subdivided into hard voting and soft voting.

Hard voting is purely atributing the majority class from the predictions of all the classifiers.

Soft voting is useful when we have probabilities as ./output/output of the classifiers, so we average those values and assume the class that it's close to the probability.


```python
"""
#soft voting
soft_vct = EnsembleVoteClassifier(clfs = classifiers[:-4], voting = 'soft')
model = soft_vct.fit(X_train,y_train)
models.append(('SoftVoter',model))
"""
```




    "\n#soft voting\nsoft_vct = EnsembleVoteClassifier(clfs = classifiers[:-4], voting = 'soft')\nmodel = soft_vct.fit(X_train,y_train)\nmodels.append(('SoftVoter',model))\n"




```python
#We are going to do just hard voting since some of the tuned classifiers can or cannot ./output/output probabilities depending on the configuration of the best parameters.

classifiers_voting = classifiers
#classifiers_voting = classifiers[:-1]

#Hard-Voting
numb_voters = np.arange(2,len(classifiers_voting)+1,1) #if catboost counted, it would be len(classifiers)+1

#we we'll try to ensemble the best two classifiers, then the best three, the best, four, etc
for i in numb_voters:
    models_names = list(copy_1.iloc[0:i,0])
    voting_models = []
    for model1 in models_names:
        for model2 in models[:-1]:
            if model2[0] == model1:
                voting_models.append(model2[1])

    hard_vct = EnsembleVoteClassifier(clfs = voting_models, voting = 'hard')
    model = hard_vct.fit(X_train,Y_train)
    name = "HardVoter_" + str(i) + "_best"
    results, fprs, tprs, labels = classify_performance(name, model, X_train, X_test, Y_train, Y_test, num_folds, scoring, seed, results, fprs, tprs, labels)
```

    HardVoter_2_best - Train: Accuracy mean - 0.883, Precision: 0.883, Recall: 0.883.
    HardVoter_2_best - Test: Accuracy - 0.881, Precision: 0.890, Recall: 0.869, F-Score: 0.880



![png](./output/output_112_1.png)



![png](./output/output_112_2.png)


    ROC AUC: 0.88
    ----------------------------------------------------------------------------------
    HardVoter_3_best - Train: Accuracy mean - 0.871, Precision: 0.874, Recall: 0.871.
    HardVoter_3_best - Test: Accuracy - 0.867, Precision: 0.897, Recall: 0.830, F-Score: 0.862



![png](./output/output_112_4.png)



![png](./output/output_112_5.png)


    ROC AUC: 0.87
    ----------------------------------------------------------------------------------
    HardVoter_4_best - Train: Accuracy mean - 0.869, Precision: 0.869, Recall: 0.869.
    HardVoter_4_best - Test: Accuracy - 0.866, Precision: 0.863, Recall: 0.869, F-Score: 0.866



![png](./output/output_112_7.png)



![png](./output/output_112_8.png)


    ROC AUC: 0.87
    ----------------------------------------------------------------------------------
    HardVoter_5_best - Train: Accuracy mean - 0.864, Precision: 0.866, Recall: 0.864.
    HardVoter_5_best - Test: Accuracy - 0.864, Precision: 0.883, Recall: 0.839, F-Score: 0.861



![png](./output/output_112_10.png)



![png](./output/output_112_11.png)


    ROC AUC: 0.86
    ----------------------------------------------------------------------------------


### Bagging (Bootstrapping Aggregation)

Bagging is characteristic of random forest. You bootstrap or subdivide the same training set into multiple subsets or bags. This steps is also called row sampling with replacement. Each of these training bags will feed a mmodel to be trained. After each model being trained, it is time for aggragation, in other orders, to predict the outcome based on a voting system of each model trained.


```python
random_forest_classifier = classifiers[0] # is the random forest classifier

bagg = BaggingClassifier(base_estimator = random_forest_classifier, verbose = 0, n_jobs = -1, random_state = seed)

model = bagg.fit(X_train, Y_train)
name = "Bagging"

results, fprs, tprs, labels = classify_performance(name, model, X_train, X_test, Y_train, Y_test, num_folds, scoring, seed, results, fprs, tprs, labels)

```

    Bagging - Train: Accuracy mean - 0.808, Precision: 0.820, Recall: 0.808.
    Bagging - Test: Accuracy - 0.828, Precision: 0.849, Recall: 0.797, F-Score: 0.822



![png](./output/output_114_1.png)



![png](./output/output_114_2.png)


    ROC AUC: 0.83
    ----------------------------------------------------------------------------------


### Blending


```python
numb_voters = np.arange(2,len(classifiers)+1,1)

#we we'll try to ensemble the best two classifiers, then the best three, the best, four, etc
for i in numb_voters:
    models_names = list(copy_2.iloc[0:i,0])
    blending_models = []
    for model1 in models_names:
        for model2 in models:
            if model2[0] == model1:
                blending_models.append(model2[1])

    blend = BlendEnsemble(n_jobs = -1, test_size = 0.5, random_state = seed)
    blend.add(blending_models)
    blend.add_meta(blending_models[0])
    model = blend.fit(X_train, Y_train)

    name = "Blending_" + str(i) + "_best"
    results, fprs, tprs, labels = classify_performance(name, model, X_train, X_test, Y_train, Y_test, num_folds, scoring, seed, results, fprs, tprs, labels)

```

    Blending_2_best - Train: Accuracy mean - 0.889, Precision: 0.890, Recall: 0.889.
    Blending_2_best - Test: Accuracy - 0.891, Precision: 0.895, Recall: 0.886, F-Score: 0.890



![png](./output/output_116_1.png)



![png](./output/output_116_2.png)


    ROC AUC: 0.89
    ----------------------------------------------------------------------------------
    Blending_3_best - Train: Accuracy mean - 0.889, Precision: 0.890, Recall: 0.889.
    Blending_3_best - Test: Accuracy - 0.891, Precision: 0.895, Recall: 0.886, F-Score: 0.890



![png](./output/output_116_4.png)



![png](./output/output_116_5.png)


    ROC AUC: 0.89
    ----------------------------------------------------------------------------------
    Blending_4_best - Train: Accuracy mean - 0.889, Precision: 0.890, Recall: 0.889.
    Blending_4_best - Test: Accuracy - 0.890, Precision: 0.891, Recall: 0.889, F-Score: 0.890



![png](./output/output_116_7.png)



![png](./output/output_116_8.png)


    ROC AUC: 0.89
    ----------------------------------------------------------------------------------
    Blending_5_best - Train: Accuracy mean - 0.889, Precision: 0.889, Recall: 0.889.
    Blending_5_best - Test: Accuracy - 0.890, Precision: 0.890, Recall: 0.890, F-Score: 0.890



![png](./output/output_116_10.png)



![png](./output/output_116_11.png)


    ROC AUC: 0.89
    ----------------------------------------------------------------------------------


### Stacking

Shortly, we are trying to fit a model upon Y_valid data and models' predictions (stacked) made during the validation stage. Then, we will try to predict from the models' predictions (stacked) of test data to see if we reach Y_test values.


```python
numb_voters = np.arange(2,len(classifiers)+1, 1)

S_train, S_test = stacking(classifiers, X_train, Y_train, X_test, regression = False, mode = 'oof_pred_bag', needs_proba = False, save_dir = None,metric = accuracy_score, n_folds = num_folds, stratified = True, shuffle = True, random_state=  seed, verbose = 0)                

super_learner = classifiers[0]
model = super_learner.fit(S_train, Y_train)
name = "Stacking_" + str(i) + "_best"
results, fprs, tprs, labels = classify_performance(name, model, S_train, S_test, Y_train, Y_test, num_folds, scoring, seed, results, fprs, tprs, labels)



```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-539-8f08e4b42f16> in <module>
          1 numb_voters = np.arange(2,len(classifiers)+1, 1)
          2
    ----> 3 S_train, S_test = stacking(classifiers, X_train, Y_train, X_test, regression = False, mode = 'oof_pred_bag', needs_proba = False, save_dir = None,metric = accuracy_score, n_folds = num_folds, stratified = True, shuffle = True, random_state=  seed, verbose = 0)
          4
          5 super_learner = classifiers[0]


    C:\Program Files\Python36\lib\site-packages\vecstack\core.py in stacking(models, X_train, y_train, X_test, sample_weight, regression, transform_target, transform_pred, mode, needs_proba, save_dir, metric, n_folds, stratified, shuffle, random_state, verbose)
        574                 # Fit 1-st level model
        575                 if mode in ['pred_bag', 'oof', 'oof_pred', 'B', 'oof_pred_bag', 'A']:
    --> 576                     _ = model_action(model, X_tr, y_tr, None, sample_weight = sample_weight_tr, action = 'fit', transform = transform_target)
        577
        578                 # Predict out-of-fold part of train set


    C:\Program Files\Python36\lib\site-packages\vecstack\core.py in model_action(model, X_train, y_train, X_test, sample_weight, action, transform)
         83             return model.fit(X_train, transformer(y_train, func = transform), sample_weight=sample_weight)
         84         else:
    ---> 85             return model.fit(X_train, transformer(y_train, func = transform))
         86     elif 'predict' == action:
         87         return transformer(model.predict(X_test), func = transform)


    ~\AppData\Roaming\Python\Python36\site-packages\xgboost\sklearn.py in fit(self, X, y, sample_weight, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set)
        545                               early_stopping_rounds=early_stopping_rounds,
        546                               evals_result=evals_result, obj=obj, feval=feval,
    --> 547                               verbose_eval=verbose, xgb_model=None)
        548
        549         self.objective = xgb_options["objective"]


    ~\AppData\Roaming\Python\Python36\site-packages\xgboost\training.py in train(params, dtrain, num_boost_round, evals, obj, feval, maximize, early_stopping_rounds, evals_result, verbose_eval, xgb_model, callbacks, learning_rates)
        202                            evals=evals,
        203                            obj=obj, feval=feval,
    --> 204                            xgb_model=xgb_model, callbacks=callbacks)
        205
        206


    ~\AppData\Roaming\Python\Python36\site-packages\xgboost\training.py in _train_internal(params, dtrain, num_boost_round, evals, obj, feval, xgb_model, callbacks)
         72         # Skip the first update if it is a recovery step.
         73         if version % 2 == 0:
    ---> 74             bst.update(dtrain, i, obj)
         75             bst.save_rabit_checkpoint()
         76             version += 1


    ~\AppData\Roaming\Python\Python36\site-packages\xgboost\core.py in update(self, dtrain, iteration, fobj)
       1019         if fobj is None:
       1020             _check_call(_LIB.XGBoosterUpdateOneIter(self.handle, ctypes.c_int(iteration),
    -> 1021                                                     dtrain.handle))
       1022         else:
       1023             pred = self.predict(dtrain)


    KeyboardInterrupt:


# Statistical tests, Comparing, Ranking, etc


```python

# Plot
fig1 = plt.figure(figsize=(15,15))
for i in range(0, len(fprs)):
    plt.plot(fprs[i], tprs[i], label=labels[i])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves All Algorithms')
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], 'k--')
plt.show()

```


![png](./output/output_120_0.png)



```python
# Precision-Recall Curve
def plot_precision_vs_recall(model, title):
    from sklearn.metrics import precision_recall_curve
    probablity = model.predict_proba(X_train)[:, 1]
    plt.figure(figsize = (18, 5))
    precision, recall, threshold = precision_recall_curve(y_train, probablity)
    plt.plot(recall, precision, 'r-', lw = 3.7)
    plt.ylabel('Recall')
    plt.xlabel('Precision')
    plt.axis([0, 1.5, 0, 1.5])
    plt.title(title)
    plt.show()

'''Now plot recall vs precision curve of rf and gbc.'''
plot_precision_vs_recall(rf, title = 'RF Precision-Recall Curve')
plot_precision_vs_recall(gbc, title = 'GBC Precision-Recall Curve')
```


```python
#Visualizing decision regions
rf_pca = RandomForestClassifier(random_state = seed)
gbc_pca = GradientBoostingClassifier(random_state = seed)
dt_pca = DecisionTreeClassifier(random_state = seed)
knn_pca = KNeighborsClassifier()
lr_pca = LogisticRegression(random_state = seed)
base_model_pca = [rf_pca, gbc_pca, dt_pca, knn_pca, lr_pca]
hard_vct_pca = EnsembleVoteClassifier(clfs = base_model_pca, voting = 'hard')

'''Function to plot decision region.'''
def plot_decision_region(model):
    from mlxtend.plotting import plot_decision_regions
    '''Train models with data pca returned. Get the train data.'''
    X = df_train_pca.values # Must be converted into numpy array.
    y = y_train.values
    model.fit(X, y)
    decision_region = plot_decision_regions(X = X, y = y, clf = model)
    plt.xlabel('PCA-1', fontsize = 15)
    plt.ylabel('PCA_2', fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    return decision_region

'''Now plot decison regions for hard voting ensemble vs base models in subplots.'''
plt.figure(figsize = (25,25))
en_models = [hard_vct_pca, rf_pca, gbc_pca, dt_pca, knn_pca, lr_pca]
en_labels = ['Hard_vct', 'RF', 'GBC', 'DT', 'KNN', 'LR']
for ax, models, labels in zip(range(1,7), en_models, en_labels):
    plt.subplot(3,2,ax)
    plot_decision_region(models)
    plt.title(labels, fontsize = 18)
plt.suptitle('Hard Voting vs Base Models Decision Regions', fontsize = 28)
plt.tight_layout(rect = [0, 0.03, 1, 0.97])
suppress_warnings()
```


```python
results.sort_values(by=["Test Accuracy"], ascending=False, inplace=True)
results.head(100)
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
      <th>Model</th>
      <th>Test Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F-Score</th>
      <th>AUC</th>
      <th>Train Accuracy</th>
      <th>Train Precision</th>
      <th>Train Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CATBOOST</td>
      <td>0.889436</td>
      <td>0.899739</td>
      <td>0.881496</td>
      <td>0.890524</td>
      <td>0.891624</td>
      <td>0.889436</td>
      <td>0.889578</td>
      <td>0.889436</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Blending_2_best</td>
      <td>0.889436</td>
      <td>0.894653</td>
      <td>0.886024</td>
      <td>0.890317</td>
      <td>0.890836</td>
      <td>0.889436</td>
      <td>0.889578</td>
      <td>0.889436</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Blending_3_best</td>
      <td>0.889436</td>
      <td>0.894653</td>
      <td>0.886024</td>
      <td>0.890317</td>
      <td>0.890836</td>
      <td>0.889436</td>
      <td>0.889578</td>
      <td>0.889436</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Blending_4_best</td>
      <td>0.889436</td>
      <td>0.890708</td>
      <td>0.888780</td>
      <td>0.889743</td>
      <td>0.889851</td>
      <td>0.889436</td>
      <td>0.889578</td>
      <td>0.889436</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Blending_5_best</td>
      <td>0.888780</td>
      <td>0.890290</td>
      <td>0.889764</td>
      <td>0.890027</td>
      <td>0.890048</td>
      <td>0.888780</td>
      <td>0.888874</td>
      <td>0.888780</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Stacking_4_best</td>
      <td>0.887631</td>
      <td>0.899270</td>
      <td>0.873425</td>
      <td>0.886159</td>
      <td>0.887786</td>
      <td>0.887631</td>
      <td>0.887696</td>
      <td>0.887631</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Stacking_3_best</td>
      <td>0.887631</td>
      <td>0.899270</td>
      <td>0.873425</td>
      <td>0.886159</td>
      <td>0.887786</td>
      <td>0.887631</td>
      <td>0.887696</td>
      <td>0.887631</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Stacking_2_best</td>
      <td>0.887631</td>
      <td>0.899270</td>
      <td>0.873425</td>
      <td>0.886159</td>
      <td>0.887786</td>
      <td>0.887631</td>
      <td>0.887696</td>
      <td>0.887631</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Stacking_5_best</td>
      <td>0.887631</td>
      <td>0.899270</td>
      <td>0.873425</td>
      <td>0.886159</td>
      <td>0.887786</td>
      <td>0.887631</td>
      <td>0.887696</td>
      <td>0.887631</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Stacked</td>
      <td>0.883202</td>
      <td>0.897278</td>
      <td>0.863189</td>
      <td>0.879904</td>
      <td>0.882175</td>
      <td>0.883202</td>
      <td>0.883418</td>
      <td>0.883202</td>
    </tr>
    <tr>
      <th>7</th>
      <td>HardVoter_2_best</td>
      <td>0.883202</td>
      <td>0.890323</td>
      <td>0.869291</td>
      <td>0.879681</td>
      <td>0.881092</td>
      <td>0.883202</td>
      <td>0.883382</td>
      <td>0.883202</td>
    </tr>
    <tr>
      <th>6</th>
      <td>XGBOOST</td>
      <td>0.883202</td>
      <td>0.889958</td>
      <td>0.874016</td>
      <td>0.881915</td>
      <td>0.882962</td>
      <td>0.883202</td>
      <td>0.883382</td>
      <td>0.883202</td>
    </tr>
    <tr>
      <th>8</th>
      <td>HardVoter_3_best</td>
      <td>0.871227</td>
      <td>0.896874</td>
      <td>0.830315</td>
      <td>0.862312</td>
      <td>0.867412</td>
      <td>0.871227</td>
      <td>0.873642</td>
      <td>0.871227</td>
    </tr>
    <tr>
      <th>9</th>
      <td>HardVoter_4_best</td>
      <td>0.869094</td>
      <td>0.862695</td>
      <td>0.869488</td>
      <td>0.866078</td>
      <td>0.865538</td>
      <td>0.869094</td>
      <td>0.869416</td>
      <td>0.869094</td>
    </tr>
    <tr>
      <th>10</th>
      <td>HardVoter_5_best</td>
      <td>0.863845</td>
      <td>0.882816</td>
      <td>0.839370</td>
      <td>0.860545</td>
      <td>0.863965</td>
      <td>0.863845</td>
      <td>0.866445</td>
      <td>0.863845</td>
    </tr>
    <tr>
      <th>11</th>
      <td>logistic</td>
      <td>0.859252</td>
      <td>0.856143</td>
      <td>0.866929</td>
      <td>0.861502</td>
      <td>0.860616</td>
      <td>0.859252</td>
      <td>0.859291</td>
      <td>0.859252</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Bagging</td>
      <td>0.807907</td>
      <td>0.849088</td>
      <td>0.797441</td>
      <td>0.822455</td>
      <td>0.827840</td>
      <td>0.807907</td>
      <td>0.819852</td>
      <td>0.807907</td>
    </tr>
    <tr>
      <th>13</th>
      <td>NB</td>
      <td>0.807743</td>
      <td>0.831671</td>
      <td>0.820866</td>
      <td>0.826233</td>
      <td>0.827346</td>
      <td>0.807743</td>
      <td>0.817834</td>
      <td>0.807743</td>
    </tr>
    <tr>
      <th>14</th>
      <td>DT</td>
      <td>0.807087</td>
      <td>0.823529</td>
      <td>0.815748</td>
      <td>0.819620</td>
      <td>0.820455</td>
      <td>0.807087</td>
      <td>0.807687</td>
      <td>0.807087</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Another way of seeing ranking
import cufflinks as cf
cf.go_offline() # Required to use plotly offline with cufflinks.
py.init_notebook_mode() # Graphs charts inline (jupyter notebook).

'''Plot bar plot.'''
figure = submission_score.iplot(kind = 'bar', asFigure = True, title = 'Models Scored on Submission', theme = 'solar')
iplot(figure)
```

## Checking overfiting/underfiting. Another Interation on Feature Selection/Reduction


```python
''Create a function that returns learning curves for different classifiers.'''
def plot_learning_curve(model):
    from sklearn.model_selection import learning_curve
    # Create feature matrix and target vector
    X, y = X_train, y_train
    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv = 10,
                                                    scoring='accuracy', n_jobs = -1,
                                                    train_sizes = np.linspace(0.01, 1.0, 17), random_state = seed)
                                                    # 17 different sizes of the training set

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis = 1)
    train_std = np.std(train_scores, axis = 1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis = 1)
    test_std = np.std(test_scores, axis = 1)

    # Draw lines
    plt.plot(train_sizes, train_mean, 'o-', color = 'red',  label = 'Training score')
    plt.plot(train_sizes, test_mean, 'o-', color = 'green', label = 'Cross-validation score')

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha = 0.1, color = 'r') # Alpha controls band transparency.
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha = 0.1, color = 'g')

    # Create plot
    font_size = 15
    plt.xlabel('Training Set Size', fontsize = font_size)
    plt.ylabel('Accuracy Score', fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.legend(loc = 'best')
    plt.grid()
```


```python
'''Now plot learning curves of the optimized models in subplots.'''
plt.figure(figsize = (25,25))
lc_models = [rf, gbc, dt, etc, abc, knn, svc, lr]
lc_labels = ['RF', 'GBC', 'DT', 'ETC', 'ABC', 'KNN', 'SVC', 'LR']

for ax, models, labels in zip (range(1,9), lc_models, lc_labels):
    plt.subplot(4,2,ax)
    plot_learning_curve(models)
    plt.title(labels, fontsize = 18)
plt.suptitle('Learning Curves of Optimized Models', fontsize = 28)
plt.tight_layout(rect = [0, 0.03, 1, 0.97])
```


```python

```

## Alternative Methods to Feature Selection

Filter Method - easiest but not the best way


```python
"""
#Filter Method - easiest but not the best way
print(X[["sex","relationship"]].corr())

X = X.drop('relationship',axis=1)
"""
```




    '\n#Filter Method - easiest but not the best way\nprint(X[["sex","relationship"]].corr())\n\nX = X.drop(\'relationship\',axis=1)\n'



### FORWARD SELECTION (wrapper method 2)

Forward selection is an iterative method in which we start with having no feature in the model. In each iteration, we keep adding the feature which best improves our model till an addition of a new variable does not improve the performance of the model.


```python
sfs1 = sfs(LinearRegression(), k_features=len(X.columns) , forward=True , scoring='r2')
sfs1 = sfs1.fit(X,Y)
fig = plot_sfs(sfs1.get_metric_dict())
plt.grid(True)
plt.show()

print(sfs1.k_feature_names_)

"""
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)

sfs1 = sfs(model_1 , k_features=9 , forward=True , scoring='r2')
sfs1 = sfs1.fit(X_train,y_train)
sfs1.k_feature_names_

lr=LinearRegression().fit(X_train,y_train)
y_pred_fs=lr.predict(X_test)
r2_score(y_test,y_pred_fs)
"""
```

### RECURSIVE FEATURE ELIMINATION (wrapper method 3)

It is a greedy optimization algorithm which aims to find the best performing feature subset. It repeatedly creates models and keeps aside the best or the worst performing feature at each iteration. It constructs the next model with the left features until all the features are exhausted. We need to find the optimum number of features,
for which the accuracy is the highest. It then ranks the features based on the order of their elimination.


```python
nof_list=np.arange(1,len(df.columns)+1)            
high_score=0

nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)
    model = LogisticRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]

print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
```


```python
#Now that we now the perfect number of features, we feed them to a RFE and get the final set of features given by RFE:'''
cols = list(X.columns)
model = LogisticRegression()

rfe = RFE(model, nof)             

X_rfe = rfe.fit_transform(X,Y)  

model.fit(X_rfe,Y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index

X = X[selected_features_rfe]

print(rfe.support_)
print(rfe.ranking_)
```

FEATURE SELECTION - EMBEDDED METHOD


```python
'''We iterate over the model training process and carefully
extract those features which contribute the most
to the training for a particular iteration,
penalizing a feature given a threshold.
Here we will do feature selection using Lasso regularization.
If the feature is irrelevant, lasso penalizes its coefficient making it 0.
Features with coefficient = 0 are removed.'''

reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
```


```python
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model");plt.show()

#Here Lasso model has taken all the features except NOX, CHAS and INDUS.
```

Example of SelectKBest (chi2)


```python
#EXAMPLE OF USING SICKIT LEARN TO SELECT THE TWO BEST FEATURES
#https://github.com/knathanieltucker/bit-of-data-science-and-scikit-learn/blob/master/notebooks/FeatureSelection.ipynb
#https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X,np.ravel(y))
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features
```

** Example of ExtRaTreesClassifier() **


```python
#FEATURE SELECTION USING EXTRA TREES CLASSIFIER
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index = X.columns)
feat_importances.nlargest(10).plot(kind = 'barh')
plt.show()
```

** Example of RFECV(RandomForestClassifier()) **


```python
#RECURSIVE FEATURE ELIMINATION with cross validation (using Random Forest)
'''Assigns weights to features (e.g., the coefficients of a linear model).
Features whose weights are the smallest are pruned out.'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

m = RFECV(RandomForestClassifier(), scoring = 'accuracy')
m.fit(X,y)
m.score(X,y)
```

Example of SelectFromModel (LinearSVC)


```python
#FEATURE SELECTION USING SelectFromModel AND SVC

'''features are considered unimportant and removed,
if the corresponding coef_ or featureimportances values
are below the provided threshold parameter.
Available heuristics are “mean”, “median”
and float multiples of these like “0.1*mean”.'''

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

m = SelectFromModel(LinearSVC(C=0.01, penalty='l1', dual=False))
m.fit(X, y)
```

** Example of SelectFromModel (LassoCV) **


```python
#FEATURE SELECTION USING SelectFromModel AND LASSO

from sklearn.linear_model import LassoCV

m = SelectFromModel(LassoCV(cv = 5))
m.fit(X,y)
#m.transform(X).shape
```




```python

```


```python

```


```python

```


```python

```


```python

```



## (Optional Advanced Optimization) Iterations of Feature Selection/Feature Creation

- Deep Feature Synthesis
    - Min, max, etc
    - Squared, Cubed, etc
- Combination of Features
- Clustering, t-SNE, etc


```python
"""
for col in X.columns:
    feature_matrix_add, feature_names_add = ft.dfs(entityset=es, target_entity = col,
                                       agg_primitives = ['min', 'max', 'mean', 'percent_true', 'all', 'any',
                                                         'sum', 'skew', 'std', range_, pcorr_],
                                       trans_primitives = ['divide'], drop_contains = list(all_features),
                                       max_depth = 2, max_features = 100,
                                       verbose = 1, n_jobs = -1,
                                       chunk_size = 1000)


    all_features += [str(x.get_name()) for x in feature_names_add]
    X = pd.concat([X, feature_matrix_add], axis = 1)
#Next it is important to do all the pre processing again befores training/testing the model
"""
```

## Save models and ML pipeline


```python

```

## Load models, to make predictions



```python

```

FULL DATA CLEANSING AND CLASSIFICATION PIPELINE¶


```python
#https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

np.random.seed(0)

# Read data from Titanic dataset.
df = pd.read_csv('https://raw.githubusercontent.com/amueller/'
               'scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv')

#We create the preprocessing pipelines for both numeric and categorical data
numeric_features = ['age','fare']

numeric_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'missing')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])

preprocessor = ColumnTransformer(transformers = [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

#Append classifier to preprocessing pipeline
#Now we have a full prediction pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='lbfgs'))])

X = df.drop('survived', axis = 1)
y = df['survived']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))
```

Format to get predictions from new data


```python
DIMENSIONALITY REDUCTION - PCA
```


```python
#EXAMPLE OF DIM REDUCTION WITH PCA
#We first scale the data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
#scaled_data.shape

#Apply PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2) #2 is the number of dimension we want
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
#scaled_data.shape

plt.figure(figsize = (8,6))
plt.scatter(x_pca[:,0], x_pca[:,1],c=df['target_col'])
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
```

A typical ML implementation for a classification problem


```python

#https://www.andreagrandi.it/2018/04/14/machine-learning-pima-indians-diabetes/
# Import all the algorithms we want to test
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor

from sklearn import model_selection

# Prepare an array with all the algorithms
models = []
models.append(('CART', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('LSVC', LinearSVC()))
models.append(('NB', GaussianNB()))
models.append(('RFC', RandomForestClassifier()))
models.append(('SVM', SVC(gamma = 'auto')))

#REGRESSION ALGORITHMS
#models.append(('LR', LogisticRegression()))
#models.append(('DTR', DecisionTreeRegressor()))
#models.append(('SGDRegressor', linear_model.SGDRegressor()))
#models.append(('BayesianRidge', linear_model.BayesianRidge()))
#models.append(('LassoLars', linear_model.LassoLars()))
#models.append(('ARDRegression', linear_model.ARDRegression()))
#models.append(('PassiveAggressiveRegressor', linear_model.PassiveAggressiveRegressor()))
#models.append(('TheilSenRegressor', linear_model.TheilSenRegressor()))
#models.append(('LinearRegression', linear_model.LinearRegression()))


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
num_folds = 3
scoring = 'accuracy'

# Prepare the configuration to run the test
seed = 7
results = []
names = []
X = train_set_scaled
y = train_set_labels

# Every algorithm is tested and results are
# collected and printed
for name, model in models:
    kfold = model_selection.KFold(
        n_splits=3, random_state=seed)
    cv_results = model_selection.cross_val_score(
        model, X, y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (
        name, cv_results.mean(), cv_results.std())
    print(msg)


# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#ONCE CHOSEN THE ALGORITHM, WE DO SCALING, PARAM GRID K FOLD AND GRID SEARCH #################
# Build a scaler
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

# Build parameter grid
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)

# Build the model
model = SVC()
kfold = KFold(n_splits = num_folds, random_state = seed)
grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = scoring, cv = kfold)
grid_result = grid.fit(rescaledX, y_train)

# Show the results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#MODEL IS NOW READY TO BE SAVED AND USED
##############################################################################################
```

Finding best parameters with GridSearchCV


```python
#https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py

from sklearn.model_selection import GridSearchCV

param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__C': [0.1, 1.0, 10, 100]
}

grid_search = GridSearchCV(clf, param_grid, cv=10, iid=False)
grid_search.fit(X_train, y_train)

print(("best logistic regression from grid search: %.3f"
       % grid_search.score(X_test, y_test)))
```


```python
#GRADIENT DESCENT = iterative technique to minimize complex loss function

'''Two ways to combat overfitting:
1. Use more training data. The more you have, the harder it is to overfit
the data by learning too much from any single training example.

2. Use regularization. Add in a penalty in the loss function for building
a model that assigns too much explanatory power to any one feature or
allows too many features to be taken into account.'''

'''Regularization loss: how much we penalize the model
for having large parameters that heavily weight certain features'''
```


```python

#Basic Keras neural network
#Source: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

# load the dataset
dataset = loadtxt(r'D:\Downloads\DATASETS\general\pima_indian_dataset.csv', delimiter=',')

# split into input (X) and ./output/output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, y, epochs=20, batch_size=40)

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
```
