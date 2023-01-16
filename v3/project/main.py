import pandas as pd

from preprocessing.Preprocessing import preprocess
from feature_engineering.Transforming import transform
from models.Models import models

# plots
import matplotlib.pyplot as plt

import eli5
from sklearn.linear_model import LogisticRegression
from IPython.display import display

# data path
DATA_PATH = 'v3/project/fake_job_postings.csv'

def plot(table):
    # split df into two [true postive, true negative][false positive, false negative]
    arr = table.values
    arr_log = [[arr[0][-4], arr[0][-1]], [arr[0][-3], arr[0][-2]]]
    arr_ran = [[arr[1][-4], arr[1][-1]], [arr[1][-3], arr[1][-2]]]
    # plot confusion matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.matshow(arr_log, cmap=plt.cm.Blues)
    ax1.set_title('Logistic Regression')
    ax1.set_xticklabels(['', 'Positive', 'Negative'])
    ax1.set_yticklabels(['', 'True', 'False'])
    ax1.text(0, 0, arr_log[0][0], va='center', ha='center')
    ax1.text(1, 0, arr_log[0][1], va='center', ha='center')
    ax1.text(0, 1, arr_log[1][0], va='center', ha='center')
    ax1.text(1, 1, arr_log[1][1], va='center', ha='center')
    
    ax2.matshow(arr_ran, cmap=plt.cm.Blues)
    ax2.set_title('Random Forest')
    ax2.set_xticklabels(['', 'Positive', 'Negative'])
    ax2.set_yticklabels(['', 'True', 'False'])
    ax2.text(0, 0, arr_ran[0][0], va='center', ha='center')
    ax2.text(1, 0, arr_ran[0][1], va='center', ha='center')
    ax2.text(0, 1, arr_ran[1][0], va='center', ha='center')
    ax2.text(1, 1, arr_ran[1][1], va='center', ha='center')
    plt.show()
    
def make_feature_weights(X, y, model):
    log_reg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                            intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs',
                            max_iter=1000, multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                            l1_ratio=None).fit(X, y)
    # save html
    html = eli5.show_weights(log_reg, feature_names=X.columns.tolist(), top=(30, 30)).data
    html_file = open('v3/project/outputs/feature_importance.html', 'w')
    html_file.write(html)
    html_file.close()
    print('Feature importance saved!')

if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test = preprocess(df)
    
    X_train = transform(X_train)
    X_test = transform(X_test)
    
    output = models(X_train, y_train, X_test, y_test)
    plot(output)
    make_feature_weights(X_train, y_train, output)
    

    
