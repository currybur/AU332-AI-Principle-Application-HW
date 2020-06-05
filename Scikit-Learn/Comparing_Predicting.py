# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import csv
import time

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer


def load_csv_data(filename):
    file = pd.read_csv(filename)

    data = file[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
                 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1',
                 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6',
                 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
                 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
                 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21',
                 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
                 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
                 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
                 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']]
    labels = file['Cover_Type']
    data = np.array(data)
    labels = np.array(labels)
    return data, labels


def load_csv_test(filename):
    file = pd.read_csv(filename)

    data = file[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
                 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1',
                 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6',
                 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
                 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
                 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21',
                 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
                 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
                 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
                 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']]
    data = np.array(data)
    return data


def predict_test(features, labels, features_test, classifier):
    rf2 = classifier

    rf2.fit(features, labels)

    predict = rf2.predict(features_test)

    # write csv
    with open("all/predict.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["Id", "Cover_Type"])
        for i in range(0, len(predict)):
            writer.writerow([(15121 + i), predict[i]])


def k_cross_validation(k, features, labels, classifier):
    sum_accuracy = 0
    k_fold = KFold(n_splits=k, random_state=5, shuffle=True)
    for train_index, test_index in k_fold.split(features):
        train_data, test_data = features[train_index], features[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        start = time.time()
        classifier.fit(train_data, train_labels)
        print("training {} model costs {} seconds.".format(str(classifier).split('(')[0], time.time()-start))
        s = classifier.score(test_data, test_labels)
        # print(s)
        sum_accuracy += s
    print("{}-fold mean accuracy: {}".format(k, sum_accuracy/k))
    return sum_accuracy/k


def default_models(features, labels):
    result = pd.DataFrame()
    for k in range(3, 14):
        lr = LogisticRegression(solver='lbfgs', multi_class='auto')
        gnb = GaussianNB()
        svc = SVC(gamma='auto')
        mlp = MLPClassifier(hidden_layer_sizes=(100,))
        dtc = DecisionTreeClassifier()
        knn = KNeighborsClassifier()
        gbc = GradientBoostingClassifier()
        forest = RandomForestClassifier(n_estimators=110)
        clfs = {"Logistic Regression": lr, "Gaussian Naive Bayes": gnb, "C-Support Vector Classification": svc, "Multi-layer Perceptron": mlp,
                "Decision Tree": dtc, "K-nearest neighbors": knn, "Gradient boosting": gbc, "Random Forest": forest}
        res = []
        for name, classifier in clfs.items():
            print(name)
            score = k_cross_validation(k, features, labels, classifier)
            res.append(score)
        result[k] = res
    result.index = ["Logistic Regression", "Gaussian Naive Bayes", "C-Support Vector Classification",
                    "Multi-layer Perceptron", "Decision Tree", "K-nearest neighbors", "Gradient boosting", "Random Forest"]
    print(result)


def grid_search(features, labels, classifier, param_test):
    grid_searcher = GridSearchCV(estimator=classifier(), param_grid=param_test, cv=8)
    grid_searcher.fit(features, labels)
    print(grid_searcher.best_estimator_)
    print(grid_searcher.best_params_)
    print(grid_searcher.best_score_)


if __name__ == '__main__':
    features, labels = load_csv_data('all/train.csv')

    # rough estimation
    default_models(features, labels)

    # different data pre-processing
    normalScaler = Normalizer().fit(features)
    standardScaler = StandardScaler().fit(features)
    binaryScaler = Binarizer().fit(features)
    normalized = normalScaler.transform(features)
    standardized = standardScaler.transform(features)
    binarized = binaryScaler.transform(features)
    default_models(normalized, labels)
    default_models(standardized, labels)
    default_models(binarized, labels)

    # tuning hyper-parameters
    param_mlp = {'hidden_layer_sized': [(i,) for i in range(100, 1000, 100)]}
    grid_search(standardized, labels, MLPClassifier, param_mlp)

    param_knn = {'n_neighbors': range(1, 8), 'weights':['uniform', 'distance']}
    grid_search(features, labels, KNeighborsClassifier, param_knn)

    param_rf = {'n_estimators': range(10, 150, 10)}
    grid_search(features, labels, RandomForestClassifier, param_rf)

    param_GB = {'n_estimators': range(80, 160, 10)}
    grid_search(features, labels, GradientBoostingClassifier, param_GB)

    # predict test data
    classifier = RandomForestClassifier(n_estimators=100)
    features_test = load_csv_test('all/test.csv')
    predict_test(features, labels, features_test, classifier)
