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
from location import process_location

# data path
DATA_PATH = r'Abgabe/../data/fake_job_postings.csv'

# read data
df = pd.read_csv(DATA_PATH)

# job_id is not needed
df.drop(columns=['job_id'], inplace=True)
drop_lines = [8850, 17233, 13541]
df.drop(drop_lines, inplace=True)

# splitting because of unbalanced data
X_train, X_test, y_train, y_test = train_test_split(df.drop('fraudulent', axis=1), 
                                                    df['fraudulent'], 
                                                    test_size=0.2, 
                                                    stratify=df['fraudulent'], 
                                                    random_state=42)

X_train = process_location(X_train)
