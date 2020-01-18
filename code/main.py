import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix


def pre_processing():
    print("PRE-PROCESSING THE DATASET...")
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

    #onehotencoder = OneHotEncoder(categories='auto')
    #features = onehotencoder.fit_transform(features).toarray()

    labelencoder_Y = LabelEncoder()
    Y = labelencoder_Y.fit_transform(Y)

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
            classifier = SVC(kernel='linear', random_state=1)
        elif classifier_name == "Neural Networks":
            classifier = Sequential()
            classifier.add(Dense(units=8, activation='relu', input_dim=14))
            classifier.add(Dense(units=8, activation='relu'))
            classifier.add(Dense(units=1, activation='sigmoid'))
            classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            #if we wanted to use scikit-learn
            #classificador = MLPClassifier(verbose=True, max_iter=1000, tol=0.000010)

        if classifier_name == "Neural Networks":
            classifier.fit(features[training_indexes], features[training_indexes], batch_size=10, epochs=3)
            predictions = classifier.predict(features[test_indexes])
            predictions = (predictions > 0.5)

        else:
            classifier.fit(features[training_indexes], Y[training_indexes])
            predictions = classifier.predict(features[test_indexes])

        precision = accuracy_score(Y[test_indexes], predictions)
        matrices.append(confusion_matrix(Y[test_indexes], predictions))
        results.append(precision)


    final_matrix = np.mean(matrices, axis=0)
    results = np.asarray(results)

    print("Confusion Matrix: ")
    print(final_matrix)
    print("Average Accuracy: " + str(results.mean()))
    print("Std Accuracy: " + str(results.std()))


def main():
    features, Y = pre_processing()
    classifiers = ["Naive Bayes", "Decision Tree", "Random Forest", "K-Nearest Neighbour", "Logistic Regression", "Support Vector Machines", "Neural Networks"]#, "Decision Tree", "Random Forest", "K-Nearest Neighbour", "Logistic Regression", "Support Vector Machines", "Neural Networks"]

    for classifier in classifiers:
        start_time = time.time()
        print("\n------------------- " + classifier + " ----------------------------")
        classify(classifier, features, Y)
        print("Time: {:2f} seconds".format((time.time() - start_time)))



if __name__ == "__main__":
    main()
