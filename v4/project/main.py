import pandas as pd

from preprocessing.Preprocessing import preprocess
from feature_engineering.Transforming import transform
from models.Models import models

from sklearn.linear_model import LogisticRegression
import eli5

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.utils import resample

# data path
DATA_PATH = 'v3/project/fake_job_postings.csv'

def plot(table, title):
    # split df into two [true postive, true negative][false positive, false negative]
    arr = table.values
    arr_log = [[arr[0][-4], arr[0][-1]], [arr[0][-2], arr[0][-3]]]
    arr_ran = [[arr[1][-4], arr[1][-1]], [arr[1][-2], arr[1][-3]]]
    cols = table.columns.values[1:5]
    metrics = table[cols].values
    metrics = pd.DataFrame(metrics, columns=cols, index=['Logistic Regression', 'Random Forest'])
    
    gs = gridspec.GridSpec(2, 2)
    plt.figure(figsize=(7, 7))
    
    ax1 = plt.subplot(gs[0, 0])
    ax1.matshow(arr_log, cmap=plt.cm.Blues)
    ax1.set_title('Logistic Regression')
    ax1.set_xticklabels(['', 'Positive', 'Negative'])
    ax1.set_yticklabels(['', 'True', 'False'])
    ax1.text(0, 0, arr_log[0][0], va='center', ha='center')
    ax1.text(1, 0, arr_log[0][1], va='center', ha='center')
    ax1.text(0, 1, arr_log[1][0], va='center', ha='center')
    ax1.text(1, 1, arr_log[1][1], va='center', ha='center')
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.matshow(arr_ran, cmap=plt.cm.Blues)
    ax2.set_title('Random Forest')
    ax2.set_xticklabels(['', 'Positive', 'Negative'])
    ax2.set_yticklabels(['', 'True', 'False'])
    ax2.text(0, 0, arr_ran[0][0], va='center', ha='center')
    ax2.text(1, 0, arr_ran[0][1], va='center', ha='center')
    ax2.text(0, 1, arr_ran[1][0], va='center', ha='center')
    ax2.text(1, 1, arr_ran[1][1], va='center', ha='center')
    
    ax3 = plt.subplot(gs[1, :])
    # plot bar chart of metrics Logistic Regression vs Random Forest
    metrics.plot(kind='bar', ax=ax3)
    ax3.set_title('Metrics')
    ax3.set_ylabel('Score')
    ax3.tick_params(axis='x', labelrotation=0)
    ax3.set_ylim(-0.0001, 1)
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%.2f')
    plt.savefig(f'v4/project/outputs/Metric_{title}.png')
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
    
    df_original = df.copy()
    X_train, X_test, y_train, y_test = preprocess(df_original)
    X_train = transform(X_train)
    X_test = transform(X_test)
    output = models(X_train, y_train, X_test, y_test)
    plot(output, 'original')
    make_feature_weights(X_train, y_train, output)
    
    
    df_60k = df.copy()
    # make fraudulent data 60k
    fraud = df_60k[df_60k['fraudulent'] == 1]
    fraud = resample(fraud, replace=True, n_samples=60000, random_state=42)
    non_fraud = df_60k[df_60k['fraudulent'] == 0]
    # concat the fraudulent and non fraudulent jobs
    df_60k = pd.concat([fraud, non_fraud])
    X_train, X_test, y_train, y_test = preprocess(df_60k)
    X_train = transform(X_train)
    X_test = transform(X_test)
    output_60k = models(X_train, y_train, X_test, y_test)
    output_60k.Model = ['Logistic Regression 60k', 'Random Forest 60k']
    # add to output
    final_output = pd.concat([output, output_60k])
    plot(final_output.iloc[2:4, :], '60k')
    

    
