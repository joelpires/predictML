import pandas as pd
import numpy as np
import time
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
from sklearn.preprocessing import Imputer
import pickle



def pre_processing(dataset="census"):
    print("PRE-PROCESSING THE DATASET...")

    if dataset == "census":
        data = pd.read_csv('../data/census.csv')

        features = data.iloc[:, 0:14].values
        Y = data.iloc[:, 14].values

        labelencoder_features = LabelEncoder()
        features[:, 1] = labelencoder_features.fit_transform(features[:, 1])
        features[:, 3] = labelencoder_features.fit_transform(features[:, 3])
        features[:, 5] = labelencoder_features.fit_transform(features[:, 5])
        features[:, 6] = labelencoder_features.fit_transform(features[:, 6])
        features[:, 7] = labelencoder_features.fit_transform(features[:, 7])
        features[:, 8] = labelencoder_features.fit_transform(features[:, 8])
        features[:, 9] = labelencoder_features.fit_transform(features[:, 9])
        features[:, 13] = labelencoder_features.fit_transform(features[:, 13])

        # onehotencoder = OneHotEncoder(categories='auto')
        # features = onehotencoder.fit_transform(features).toarray()

        labelencoder_Y = LabelEncoder()
        Y = labelencoder_Y.fit_transform(Y)

    if dataset == "credit_data":
        data = pd.read_csv('../data/credit_data.csv')

        data.loc[data.age < 0, 'age'] = 40.92

        features = data.iloc[:, 1:4].values
        Y = data.iloc[:, 4].values

        imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imputer = imputer.fit(features[:, 1:4])
        features[:, 1:4] = imputer.transform(features[:, 1:4])


    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return (features, Y)


def classify(classifier_name, features, Y):
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


        classifier.fit(features[training_indexes], Y[training_indexes])
        predictions = classifier.predict(features[test_indexes])

        precision = accuracy_score(Y[test_indexes], predictions)
        matrices.append(confusion_matrix(Y[test_indexes], predictions))
        results.append(precision)

    return (classifier, matrices, results)

def evaluate(matrices=None, results=None):

    final_matrix = np.mean(matrices, axis=0)
    aux = np.asarray(results)
    acccuracy = aux.mean()    #Average

    print("Std Accuracy: " + str(aux.std()))
    print("Average Accuracy: " + str(acccuracy))
    print("Confusion Matrix: ")
    print(final_matrix)



def dump_model(dataset, classifier, classifier_name):
    filename = "../models/" + dataset + "/" + classifier_name + ".sav"
    pickle.dump(classifier, open(filename, 'wb'))



def load_model(dataset, classifier_name):
    filename = "../models/" + dataset + "/" + classifier_name + ".sav"
    pickle.load(open(filename, 'rb'))


def predict(classifier, new_features):

    scaler = StandardScaler()
    new_features = np.asarray(new_features)
    new_features = new_features.reshape(-1, 1)
    new_features = scaler.fit_transform(new_features)
    new_features = new_features.reshape(-1, 3)

    return classifier.predict(new_features)


def ensemble_classification(classifier, new_features):


def main():
    task = "classify"
    dataset = ""

    if task == "classify":
        dataset = ["census"]
        algorithms = ["Naive Bayes", "Decision Tree", "Random Forest", "K-Nearest Neighbour", "Logistic Regression", "Support Vector Machines", "Neural Networks"]
    else:
        dataset = ["house_prices"]
        algorithms = ["Simple Regression", "Polinomial Regression", "Random Forest", "Support Vector Machines", "Neural Networks"]

    features, Y = pre_processing(dataset)

    ranking = True
    laps = 1
    flag = "build_model"
    new_features = []

    for algorith_name in algorithms:
        start_time = time.time()
        print("\n------------------- " + algorith_name + " ----------------------------")
        classifier = None

        if flag == "load_model":
            classifier = load_model(dataset, algorith_name)
            predict(classifier, new_features)
        elif flag == "build_model":
            if ranking:
                laps = 30
            for i in laps:
                classifier, matrices, results = classify(algorith_name, features, Y)
            dump_model(dataset, classifier, algorith_name)
            evaluate(matrices, results)

        print("Time: {:2f} seconds".format((time.time() - start_time)))



if __name__ == "__main__":
    main()
