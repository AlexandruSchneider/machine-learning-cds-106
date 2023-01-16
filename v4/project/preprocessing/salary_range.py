import pandas as pd
def process_salary_range(df):

    # filter dates
    dates = ['Dec', 'Apr', 'Oct', 'Sep', 'Jun', 'Nov']
    df['salary_range'].replace(dates, 'Date', regex=True, inplace=True)

    # filter numbers (remove $ and other measures)
    replace_with_nothing = ['$', ',', ' ']
    df['salary_range'].replace(replace_with_nothing, '', regex=True, inplace=True)
    
    # remove everything and replace by ' ' (space) to process later
    replace_to_process = ['-', 'to', 'Date']
    df['salary_range'].replace(replace_to_process, ' ', regex=True, inplace=True)
    
    # if there is a ' ' (space), select the smaller (first) value
    df['salary_range'] = df['salary_range'].str.split(' ').str[0]
    
    # conert datatype into numbers
    df['salary_range'] = pd.to_numeric(df['salary_range'], errors='coerce')
    
    # split into 4 qurtiles and label them
    df['salary_range'] = pd.qcut(df['salary_range'], q=4, labels=['low', 'medium', 'high', 'very high'])
    
    # add category 'Unspecified' and fill missing values with it
    df['salary_range'] = df['salary_range'].cat.add_categories('Unspecified')
    df['salary_range'] = df['salary_range'].fillna('Unspecified')
    df['salary_range'] = df['salary_range'].astype(str)
    # output that preprocessing is done
    print('Preprocessing salary_range DONE!')

    return df
