import pandas as pd

from preprocessing.Preprocessing import preprocess
from feature_engineering.Transforming import transform
from models.Models import models


# data path
DATA_PATH = r'v2/fake_job_postings.csv'

if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess(df)
    X_train = transform(X_train)
    # Louis TODO: Change varname of models and output
    print(X_train.head(3))
    output = models()
    