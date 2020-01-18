import pandas as pd
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

    #labelencoder_Y = LabelEncoder()
    #Y = labelencoder_class.fit_transform(Y)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return (features, Y)


def train_test_validation(features, Y,  _test_size=0.15,_random_state=0):
    print("VALIDATION...")
    X_train, X_test, Y_train, Y_test = train_test_split(features, Y,  test_size=_test_size, random_state=_random_state)
    return (X_train, X_test, Y_train, Y_test)


def classify(classifier, X_train, X_test, Y_train):
    print("CLASSIFYING...")
    if classifier == "Naive Bayes":
        classifier = GaussianNB()
    elif classifier == "Decision Tree":
        classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    elif classifier == "Random Forest":
        classifier = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
    elif classifier == "K-Nearest Neighbour":
        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    elif classifier == "Logistic Regression":
        classifier = LogisticRegression()
    elif classifier == "Support Vector Machines":
        classifier = SVC(kernel='linear', random_state=1)

    classifier.fit(X_train, Y_train)
    predictions = classifier.predict(X_test)
    return predictions


def evaluate(predictions, Y_test):
    print("EVALUATING...")
    precision = accuracy_score(Y_test, predictions)
    matrix = confusion_matrix(Y_test, predictions)
    print("Precision: " + str(precision))
    print("Confusion Matrix: ")
    print(str(matrix))


def main():
    features, Y = pre_processing()
    X_train, X_test, Y_train, Y_test = train_test_validation(features, Y)
    classifiers = ["Naive Bayes", "Decision Tree", "Random Forest", "K-Nearest Neighbour", "Logistic Regression", "Support Vector Machines"]

    for classifier in classifiers:
        start_time = time.time()
        print("\n------------------- " + classifier + " ----------------------------")
        predictions = classify(classifier, X_train, X_test, Y_train)
        evaluate(predictions, Y_test)
        print("Time: {:2f} seconds".format((time.time() - start_time)))



if __name__ == "__main__":
    main()
