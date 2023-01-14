import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split

# Plots
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import feature_engine.imputation as imp


# preprocess the location (own class)
from .location import process_location
from .salary_range import process_salary_range

def preprocess(df):
    print(f'-------------------------------------\nStarting Preprocessing\n-------------------------------------')
    # job_id is not needed
    df.drop(columns=['job_id'], inplace=True)
    drop_lines = [8850, 17233, 13541]
    df.drop(drop_lines, inplace=True)
    # dropna from location
    df.dropna(subset=['location'], inplace=True)

    # splitting because of unbalanced data
    X_train, X_test, y_train, y_test = train_test_split(df.drop('fraudulent', axis=1), 
                                                        df['fraudulent'], 
                                                        test_size=0.2, 
                                                        stratify=df['fraudulent'], 
                                                        random_state=42)
    # y_train has to be passed, as there are additional rows generated if there are multiple cities given
    X_train, y_train = process_location(X_train, y_train)
    X_train = process_salary_range(X_train)

    # define text features to remove
    text_features = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    X_train.drop(columns=text_features, inplace=True)
    print(f'-------------------------------------\nEnd Preprocessing\n-------------------------------------')
    return X_train, X_test, y_train, y_test
