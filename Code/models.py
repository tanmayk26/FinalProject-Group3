# Import required libraries

# Models
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score


def models(X_train, X_test, y_train, y_test):

    # 1. Logistic Regression

    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print("Logistic Regression: ", accuracy_score(y_test, y_pred))
    print("Logistic Regression: \n", confusion_matrix(y_test, y_pred))
    print("Logistic Regression: \n", classification_report(y_test, y_pred))
    f1scorelr = f1_score(y_test, y_pred, average='weighted')
    print(" F1 Score of Logistic Regression Model is", f1scorelr)


    # 2. Decision Tree Classifier

    # Create an instance
    classifier = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                           sampling_strategy='not majority',
                                           replacement=False,
                                           random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("Decision Tree Classifier: ", accuracy_score(y_test, y_pred))
    print("Decision Tree Classifier: \n", confusion_matrix(y_test, y_pred))
    print("Decision Tree Classifier: \n", classification_report(y_test, y_pred))
    f1scoreknn = f1_score(y_test, y_pred, average='weighted')
    print(" F1 Score of Decision Tree Classifier Model is", f1scoreknn)

    # 3. KNN

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("KNN: ", accuracy_score(y_test, y_pred))
    print("KNN: \n", confusion_matrix(y_test, y_pred))
    print("KNN: \n", classification_report(y_test, y_pred))
    f1scoreknn = f1_score(y_test, y_pred, average='weighted')
    print(" F1 Score of KNN Model is", f1scoreknn)

    # # 4. SVC
    #
    # classifier = SVC(kernel='rbf', random_state=0)
    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    # print("SVM: ", accuracy_score(y_pred, y_test))
    # print("SVM: \n", confusion_matrix(y_pred, y_test))
    # print("SVM: \n", classification_report(y_pred, y_test))
    # f1scoresvm = f1_score(y_test, y_pred, average='weighted')
    # print(" F1 Score of SVM Model is", f1scoresvm)


    # 5. Decision Tree Classifier

    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("Decision Tree: ", accuracy_score(y_pred, y_test))
    print("Decision Tree: \n", confusion_matrix(y_pred, y_test))
    print("Decision Tree: \n", classification_report(y_pred, y_test))
    f1scoredt = f1_score(y_test, y_pred, average='weighted')
    print(" F1 Score of Decision Tree Model is", f1scoredt)

    # 6. Random Forest Classifier

    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("Random Forest: ", accuracy_score(y_test, y_pred))
    print("Random Forest: \n", confusion_matrix(y_test, y_pred))
    print("Random Forest: \n", classification_report(y_test, y_pred))
    f1scorerf = f1_score(y_test, y_pred, average='weighted')
    print(" F1 Score of Decision Tree Model is", f1scorerf)


    # 7. Gaussian

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("Gaussian NB: ", accuracy_score(y_test, y_pred))
    print("Gaussian NB: \n", confusion_matrix(y_test, y_pred))
    print("Gaussian NB: \n", classification_report(y_test, y_pred))
    f1scoregn = f1_score(y_test, y_pred, average='weighted')
    print(" F1 Score of Gaussian NB Model is", f1scoregn)


    # 8. Ridge Classifier

    classifier = RidgeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("RidgeClassifier: ", accuracy_score(y_test, y_pred))
    print("RidgeClassifier: \n", confusion_matrix(y_test, y_pred))
    print("RidgeClassifier: \n", classification_report(y_test, y_pred))
    f1scorerc = f1_score(y_test, y_pred, average='weighted')
    print(" F1 Score of Ridge Regression Model is", f1scorerc)

    # 9. MLP Classifier


    classifier = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, alpha=1e-4,
                               solver='sgd', verbose=10, random_state=42,
                               learning_rate_init=.1)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("MLPClassifier: ", accuracy_score(y_test, y_pred))
    print("MLPClassifier: \n", confusion_matrix(y_test, y_pred))
    print("MLPlassifier: \n", classification_report(y_test, y_pred))
    f1scoremlp = f1_score(y_test, y_pred, average='weighted')
    print(" F1 Score of MLP Model is", f1scoremlp)
