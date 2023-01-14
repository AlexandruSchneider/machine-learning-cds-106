import pandas as pd

from preprocessing.Preprocessing import preprocess
from feature_engineering.Transforming import transform
from models.Models import models


# data path
DATA_PATH = 'fake_job_postings.csv'

if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)

    # oversampling
    # data_1f = df[df.fraudulent == 1]
    # df = pd.concat([df] + [data_1f] * 7, axis=0)

    X_train, X_test, y_train, y_test = preprocess(df)
    X_train = transform(X_train)
    X_test = transform(X_test)

    output = models(X_train, y_train, X_test, y_test)
    print(output)
    print('DONE!')
