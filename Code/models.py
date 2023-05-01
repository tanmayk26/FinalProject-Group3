from tools import plot_confusion_matrix

import pandas as pd
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score


def models(X_train, X_test, y_train, y_test, labels):
    results = []
    print('Building Models...')

    # 1. Logistic Regression
    method_name = 'Logistic Regression'
    print(f'Building {method_name} model...')
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    # print("Logistic Regression Accuracy: ", lg_accuracy)
    # print("Logistic Regression ROC AUC: ", lg_roc)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels, method_name)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    results.append([method_name, accuracy, roc, precision, recall, f1_score])

    # # 2. Decision Tree Classifier
    # classifier = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
    #                                        sampling_strategy='not majority',
    #                                        replacement=False,
    #                                        random_state=42)
    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    # y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    # dt_accuracy = accuracy_score(y_test, y_pred)
    # dt_roc = roc_auc_score(y_test, y_pred_proba)
    # print("Decision Tree Accuracy: ", dt_accuracy)
    # print("Decision Tree ROC AUC: ", dt_roc)
    # cm = confusion_matrix(y_test, y_pred)
    # plot_confusion_matrix(cm, labels)
    # results.append(['Decision Tree', dt_accuracy, dt_roc])

    # 3. KNN
    method_name = 'KNN'
    print(f'Building {method_name} model...')
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    # print("KNN Accuracy: ", knn_accuracy)
    # print("KNN ROC AUC: ", knn_roc)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels, method_name)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    results.append([method_name, accuracy, roc, precision, recall, f1_score])

    # 5. Decision Tree Classifier
    method_name = 'Decision Tree Classifier'
    print(f'Building {method_name} model...')
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    # print("Decision Tree Accuracy: ", dtt_accuracy)
    # print("Decision Tree ROC AUC: ", dtt_roc)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels, method_name)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    results.append([method_name, accuracy, roc, precision, recall, f1_score])

    # 6. Random Forest Classifier
    method_name = 'Random Forest Classifier'
    print(f'Building {method_name} model...')
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    # print("Random Forest Accuracy: ", rf_accuracy)
    # print("Random Forest ROC AUC: ", rf_roc)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels, method_name)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    results.append([method_name, accuracy, roc, precision, recall, f1_score])

    # 7. Gaussian
    method_name = 'Naive Bayes'
    print(f'Building {method_name} model...')
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    # print("Naive Bayes Accuracy: ", nb_accuracy)
    # print("Naive Bayes ROC AUC: ", nb_roc)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels, method_name)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    results.append([method_name, accuracy, roc, precision, recall, f1_score])

    # # 8. Ridge Classifier
    # method_name = 'Ridge Classifier'
    # print(f'Building {method_name} model...')
    # classifier = RidgeClassifier()
    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    # # RidgeClassifier does not have predict_proba method, so ROC AUC cannot be calculated directly
    # rc_accuracy = accuracy_score(y_test, y_pred)
    #
    # print("Ridge Classifier Accuracy: ", rc_accuracy)
    # print("Ridge Classifier ROC AUC: ", "NA")
    # cm = confusion_matrix(y_test, y_pred)
    # plot_confusion_matrix(cm, labels, method_name)
    # results.append(['Ridge Classifier ', rc_accuracy, "NA"])

    # 9. MLP Classifier
    method_name = 'MLP Classifier'
    print(f'Building {method_name} model...')
    classifier = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, alpha=1e-4,
                               solver='sgd', verbose=10, random_state=42,
                               learning_rate_init=.1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    # print("MLP Classifier Accuracy: ", mlp_accuracy)
    # print("MLP Classifier ROC AUC: ", mlp_roc)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels, method_name)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    results.append([method_name, accuracy, roc, precision, recall, f1_score])

    # 10. XGB Classifier
    method_name = 'XGB Classifier'
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    # print("XGB Classifier Accuracy: ", mlp_accuracy)
    # print("XGB Classifier ROC AUC: ", mlp_roc)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels, method_name)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    results.append([method_name, accuracy, roc, precision, recall, f1_score])

    results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'ROC AUC', 'Precision', 'Recall', 'F1-score'])
    print(results_df)


