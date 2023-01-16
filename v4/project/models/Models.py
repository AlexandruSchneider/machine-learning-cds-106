from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from imblearn.over_sampling import SMOTE


def plot_roc_curve(y_test, y_pred, model_name):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve of:  ' + model_name)
    plt.legend(loc="lower right")
    plt.savefig('v4/project/outputs/' + model_name + '.png')
    plt.show()

def logisticRegression(X_train, y_train, X_test, y_test):
    # hyperparameter
    model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                               intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs',
                               max_iter=1000, multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                               l1_ratio=None)
    
    model.fit(X_train, y_train)
    smote = SMOTE(random_state=0)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    y_pred = model.predict(X_test)
    plot_roc_curve(y_test, y_pred, 'Logistic Regression')

    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), \
           recall_score(y_test, y_pred), confusion_matrix(y_test, y_pred)[0][0], confusion_matrix(y_test, y_pred)[0][1], \
           confusion_matrix(y_test, y_pred)[1][0], confusion_matrix(y_test, y_pred)[1][1]


def randomForest(X_train, y_train, X_test, y_test):
    # hyperparameter
    model = RandomForestClassifier(n_estimators=100,
                                   criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None,
                                   min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None,
                                   random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0,
                                   max_samples=None)
    
    model.fit(X_train, y_train)
    smote = SMOTE(random_state=0)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    y_pred = model.predict(X_test)
    plot_roc_curve(y_test, y_pred, 'Random Forest')

    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), \
           recall_score(y_test, y_pred), confusion_matrix(y_test, y_pred)[0][0], confusion_matrix(y_test, y_pred)[0][1], \
           confusion_matrix(y_test, y_pred)[1][0], confusion_matrix(y_test, y_pred)[1][1]


def models(X_train, y_train, X_test, y_test):
    print(f'-------------------------------------\nStarting Modelling Process\n-------------------------------------')
    results = pd.DataFrame(columns=
                           ['Model', 'Accuracy', 'F1 Score', 'Precision',
                            'Recall',
                            'True Positive', 'False Positive', 'False Negative', 'True Negative'
                            ])

    lr_list = logisticRegression(X_train, y_train, X_test, y_test)
    print(f'Logistic Regression DONE!')
    results = results.append(
        {'Model': 'Logistic Regression', 'Accuracy': lr_list[0], 'F1 Score': lr_list[1], 'Precision': lr_list[2],
         'Recall': lr_list[3],
         'True Positive': lr_list[4], 'False Positive': lr_list[5],
         'False Negative': lr_list[6], 'True Negative': lr_list[7]
         }, ignore_index=True)

    rf_list = randomForest(X_train, y_train, X_test, y_test)
    print(f'Random Forest DONE!')
    results = results.append(
        {'Model': 'Random Forest', 'Accuracy': rf_list[0], 'F1 Score': rf_list[1], 'Precision': rf_list[2],
         'Recall': rf_list[3],
         'True Positive': rf_list[4], 'False Positive': rf_list[5],
         'False Negative': rf_list[6], 'True Negative': rf_list[7]
         }, ignore_index=True)
    print(f'-------------------------------------\nEnd Modelling Process\n-------------------------------------')
    return results
