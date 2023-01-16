import numpy as np
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline



def transform(df):
    print('-------------------------------------\nStarting Transforming\n-------------------------------------')    
    one_hot_features = ['salary_range', 'employment_type', 'required_experience', 'required_education']
    ordinal_features = ['country', 'state', 'city', 'department', 'industry', 'function']
    encoder_pipeline = Pipeline([
        ('onehot', OneHotEncoder(top_categories=None, variables=one_hot_features, drop_last=True)),
        ('ordinal_enc', OrdinalEncoder(encoding_method='arbitrary', variables=ordinal_features))
    ])
    print(f'Pipeline created successfully')

    encoder_pipeline.fit(df)
    print(f'Pipeline fitted successfully')
    df = encoder_pipeline.transform(df)
    print(f'Pipeline transformed successfully')
    df = df.astype(np.int64)
    print('-------------------------------------\nEnd Transforming\n-------------------------------------')
    return df