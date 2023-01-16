from feature_engine.imputation import AddMissingIndicator, MeanMedianImputer, CategoricalImputer
from sklearn.pipeline import Pipeline

def imput_missing_values(df):
    imputer = Pipeline([
        ('missing_indicator', AddMissingIndicator()),
        ('mean_median_imputer', MeanMedianImputer(imputation_method='median')),
        ('categorical_imputer', CategoricalImputer(imputation_method='frequent', variables=['department', 
                                                                                      'employment_type', 
                                                                                      'required_experience',
                                                                                      'required_education',
                                                                                      'industry',
                                                                                      'function']))
    ])
    
    df = imputer.fit_transform(df)
    print(f'Preprocessing imputation DONE!')
    
    return df