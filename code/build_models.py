import pandas as pd
import numpy as np
import time
import pickle
from sklearn.preprocessing import Imputer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from mlxtend.preprocessing import DenseTransformer
from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings('ignore')



def pre_processing(dataset="census"):
    print("PRE-PROCESSING THE DATASET...")


    data = pd.read_csv('../data/census_original.csv')
    data = data[(data != '?').all(axis=1)]

    features = data.drop('income', axis=1).values

    if dataset == "census":
        """
        numeric_features = ['age', 'hour-per-week']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler(with_mean=False))])

        categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'race', 'sex', 'native-country']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ('scaler', StandardScaler(with_mean=False))])

        preprocesser = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])
 
        preprocess = make_column_transformer(
            (['age', 'hour-per-week'], StandardScaler()),
            (['age', 'hour-per-week'], SimpleImputer(strategy='median')),
            (['workclass', 'education', 'marital-status', 'occupation', 'race', 'sex', 'native-country'], OneHotEncoder()),
            (['workclass', 'education', 'marital-status', 'occupation', 'race', 'sex', 'native-country'], StandardScaler(with_mean=False))
        )
        """
        labelencoder_previsores = LabelEncoder()
        features[:, 1] = labelencoder_previsores.fit_transform(features[:, 1])
        features[:, 3] = labelencoder_previsores.fit_transform(features[:, 3])
        features[:, 5] = labelencoder_previsores.fit_transform(features[:, 5])
        features[:, 6] = labelencoder_previsores.fit_transform(features[:, 6])
        features[:, 7] = labelencoder_previsores.fit_transform(features[:, 7])
        features[:, 8] = labelencoder_previsores.fit_transform(features[:, 8])
        features[:, 9] = labelencoder_previsores.fit_transform(features[:, 9])
        features[:, 13] = labelencoder_previsores.fit_transform(features[:, 13])

        scaler = StandardScaler()
        features = scaler.fit_transform(features)




    Y = data.income.values
    labelencoder_classe = LabelEncoder()
    Y = labelencoder_classe.fit_transform(Y)


    preprocesser = None
    return (preprocesser, features, Y)


def classify(classifier_name, preprocesser, features, Y):
    print("CLASSIFYING...")

    classifier = None
    predictions = None

    results = []
    matrices = []

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)

    for training_indexes, test_indexes in kfold.split(features, np.zeros(shape=(features.shape[0], 1))):

        if classifier_name == "Naive Bayes":
            classifier = GaussianNB()
        elif classifier_name == "Decision Tree":
            classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
        elif classifier_name == "Random Forest":
            classifier = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
        elif classifier_name == "K-Nearest Neighbour":
            classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        elif classifier_name == "Logistic Regression":
            classifier = LogisticRegression()
        elif classifier_name == "Support Vector Machines":
            classifier = SVC(kernel = 'rbf', C = 2.0, random_state=1)
        elif classifier_name == "Neural Networks":
            """
            #if we wanted to use keras/tensorflow
            classifier = Sequential()
            classifier.add(Dense(units=8, activation='relu', input_dim=14))
            classifier.add(Dense(units=8, activation='relu'))
            classifier.add(Dense(units=1, activation='sigmoid'))
            classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            classifier.fit(features[training_indexes], Y[training_indexes], batch_size=10, epochs=3)
            predictions = classifier.predict(features[test_indexes])
            predictions = (predictions > 0.5)
            """

            classifier = MLPClassifier(verbose = True, max_iter = 100,
                                 tol = 0.000010, solver = 'adam',
                                 hidden_layer_sizes=(100), activation = 'relu',
                                 batch_size = 200, learning_rate_init = 0.001)

        X_train = features[training_indexes]
        Y_train = Y[training_indexes]
        X_test = features[test_indexes]
        Y_test = Y[test_indexes]

        pipe = None
        #pipe = make_pipeline(preprocesser, DenseTransformer(), classifier)

        classifier.fit(X_train, Y_train)
        predictions = classifier.predict(X_test)

        precision = accuracy_score(Y_test, predictions)
        matrices.append(confusion_matrix(Y_test, predictions))
        results.append(precision)

    return (pipe, matrices, results)

def evaluate(matrices=None, results=None):

    final_matrix = np.mean(matrices, axis=0)
    aux = np.asarray(results)
    acccuracy = aux.mean()    #Average

    print("Std Accuracy: " + str(aux.std()))
    print("Average Accuracy: " + str(acccuracy))
    print("Confusion Matrix: ")
    print(final_matrix)



def dump_model(dataset, classifier_name, pipe):
    filename = "../models/" + dataset + "/" + classifier_name + ".sav"
    pickle.dump(pipe, open(filename, 'wb'))


#def ensemble_classification(classifier, new_features):


def main():
    task = "classify"

    if task == "classify":
        dataset = "census"
        algorithms = ["Naive Bayes", "Decision Tree", "Random Forest", "K-Nearest Neighbour", "Logistic Regression", "Support Vector Machines", "Neural Networks"]
    else:
        dataset = "house_prices"
        algorithms = ["Simple Regression", "Polinomial Regression", "Random Forest", "Support Vector Machines", "Neural Networks"]

    preprocesser, features, Y = pre_processing(dataset)


    for algorith_name in algorithms:
        start_time = time.time()
        print("\n------------------- " + algorith_name + " ----------------------------")
        pipe, matrices, results = classify(algorith_name, preprocesser, features, Y)
        dump_model(dataset, algorith_name, pipe)
        evaluate(matrices, results)
        print("Time: {:2f} seconds".format((time.time() - start_time)))



if __name__ == "__main__":
    main()
