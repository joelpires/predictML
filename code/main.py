import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def pre_processing():
    base = pd.read_csv('../data/census.csv')

    previsores = base.iloc[:, 0:14].values
    classe = base.iloc[:, 14].values

    labelencoder_previsores = LabelEncoder()
    previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
    previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
    previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
    previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
    previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
    previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
    previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
    previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])

    #onehotencoder = OneHotEncoder(categories='auto')
    #previsores = onehotencoder.fit_transform(previsores).toarray()

    #labelencoder_classe = LabelEncoder()
    #classe = labelencoder_classe.fit_transform(classe)

    scaler = StandardScaler()
    previsores = scaler.fit_transform(previsores)

    return (previsores, classe)


def train_test_validation(previsores, classe, _test_size=0.15,_random_state=0):
    previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=_test_size, random_state=_random_state)
    return (previsores_treinamento, previsores_teste, classe_treinamento, classe_teste)


def classify(classificador, previsores_treinamento, previsores_teste, classe_treinamento):

    if classificador == "Naive Bayes":
        classificador = GaussianNB()
    elif classificador == "Decision Tree":
        classificador = DecisionTreeClassifier(criterion='entropy', random_state=0)
    elif classificador == "Random Forest":
        classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
    elif classificador == "K-Nearest Neighbour":
        classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

    classificador.fit(previsores_treinamento, classe_treinamento)
    previsoes = classificador.predict(previsores_teste)
    return previsoes


def evaluate(previsoes, classe_teste):
    precisao = accuracy_score(classe_teste, previsoes)
    matriz = confusion_matrix(classe_teste, previsoes)
    print("Precisao: " + str(precisao))
    print("Matriz: " + str(matriz))


def main():
    previsores, classe = pre_processing()
    previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_validation(previsores, classe)
    classifiers = ["Naive Bayes", "Decision Tree", "Random Forest", "K-Nearest Neighbour"]

    for classifier in classifiers:
        previsoes = classify(classifier, previsores_treinamento, previsores_teste, classe_treinamento)
        evaluate(previsoes, classe_teste)



if __name__ == "__main__":
    main()