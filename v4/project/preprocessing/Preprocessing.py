from sklearn.model_selection import train_test_split

# preprocess the location (own class)
from .location import process_location
from .salary_range import process_salary_range
from .imputation import imput_missing_values

def preprocess(df):
    print(f'-------------------------------------\nStarting Preprocessing\n-------------------------------------')
    # job_id is not needed
    # define text features to remove
    text_features = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    drop_lines = [8850, 17233, 13541]
    # drop job_id and unnecessary lines
    df.drop(columns=['job_id'], inplace=True)
    df.drop(drop_lines, inplace=True)
    # filna 'Unspecified' for location (country, state, city)
    df.location.fillna('Unspecified, Unspecified, Unspecified', inplace=True)

    # splitting because of unbalanced data
    X_train, X_test, y_train, y_test = train_test_split(df.drop('fraudulent', axis=1), 
                                                        df['fraudulent'], 
                                                        test_size=0.2, 
                                                        stratify=df['fraudulent'], 
                                                        random_state=42)
    # y_train has to be passed, as there are additional rows generated if there are multiple cities given
    X_train, y_train = process_location(X_train, y_train)
    X_train = process_salary_range(X_train)
    X_train.drop(columns=text_features, inplace=True)
    X_train = imput_missing_values(X_train)
    
    X_test, y_test = process_location(X_test, y_test)
    X_test = process_salary_range(X_test)
    X_test.drop(columns=text_features, inplace=True)
    X_test = imput_missing_values(X_test)
    
    print(f'-------------------------------------\nEnd Preprocessing\n-------------------------------------')
    return X_train, X_test, y_train, y_test