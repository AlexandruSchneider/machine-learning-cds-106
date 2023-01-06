import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "./data/fake_job_postings.csv"
postings = pd.read_csv(DATA_PATH)

df = postings



'''split df into two dataframes, one with salary_range and one without'''
df_salary = df[df['salary_range'].notnull()]
df_no_salary = df[df['salary_range'].isnull()]
# print(df_salary['salary_range'].value_counts())
# print(df_no_salary['salary_range'].value_counts())

#%%


'''make a copy of df_salary'''
df_salary_copy = df_salary.copy()

#%%
df_salary_copy['salary_range'] = df_salary_copy['salary_range'].str.replace('Dec', 'Date')
df_salary_copy['salary_range'] = df_salary_copy['salary_range'].str.replace('Apr', 'Date')
df_salary_copy['salary_range'] = df_salary_copy['salary_range'].str.replace('Oct', 'Date')
df_salary_copy['salary_range'] = df_salary_copy['salary_range'].str.replace('Sep', 'Date')
df_salary_copy['salary_range'] = df_salary_copy['salary_range'].str.replace('Jun', 'Date')
df_salary_copy['salary_range'] = df_salary_copy['salary_range'].str.replace('Nov', 'Date')



'''how many percent of salary_range intails 'date'  '''
print(df_salary_copy['salary_range'].str.contains('Date').sum() / len(df_salary_copy['salary_range']))

#%%
''''make new df with only salary range 'Date' '''
df_salary_copy_date = df_salary_copy[df_salary_copy['salary_range'].str.contains('Date')]
print(df_salary_copy_date['salary_range'].value_counts())
#%%
'''how many percent of salary_range intails 'date' and are fraudulent  '''
print(df_salary_copy[df_salary_copy['fraudulent'] == 1]['salary_range'].str.contains('Date').sum() / len(df_salary_copy[df_salary_copy['fraudulent'] == 1]['salary_range']))


#%%
'''create dataframe with only rows that have a salary_range that does include 'date' '''
df_salary_date = df_salary_copy[df_salary_copy['salary_range'].str.contains('Date') == True]

'''how many percent df_salary_date are fraudulent'''

print(df_salary_date['fraudulent'].sum() / len(df_salary_date['fraudulent']))
#%%
'''remove all non numeric characters from salary_range'''

'''make a copy of df_salary'''
df_salary_copy_numeric = df_salary_copy.copy()

df_salary_copy_numeric['salary_range'] = df_salary_copy_numeric['salary_range'].str.replace('$', '')
df_salary_copy_numeric['salary_range'] = df_salary_copy_numeric['salary_range'].str.replace(',', '')
df_salary_copy_numeric['salary_range'] = df_salary_copy_numeric['salary_range'].str.replace('K', '')
df_salary_copy_numeric['salary_range'] = df_salary_copy_numeric['salary_range'].str.replace('k', '')
df_salary_copy_numeric['salary_range'] = df_salary_copy_numeric['salary_range'].str.replace(' ', '')
df_salary_copy_numeric['salary_range'] = df_salary_copy_numeric['salary_range'].str.replace('-', ' ')
df_salary_copy_numeric['salary_range'] = df_salary_copy_numeric['salary_range'].str.replace('to', ' ')



df_salary_copy_numeric['salary_range'] = df_salary_copy_numeric['salary_range'].str.replace('Date', ' ')



#%%
'''if there are 2 numbers in salary_range, take the average'''
for index, row in df_salary_copy_numeric.iterrows():
    if len(row['salary_range'].split()) == 2:
        df_salary_copy_numeric.at[index, 'salary_range'] = (int(row['salary_range'].split()[0]) + int(row['salary_range'].split()[1])) / 2
    else:
        df_salary_copy_numeric.at[index, 'salary_range'] = row['salary_range']



#print(df_salary_copy_numeric['salary_range'].unique())
#print(len(df_salary_copy_numeric['salary_range'].unique()))

print(df_salary_copy_numeric['salary_range'].value_counts())

#%%

'''convert salary_range to numeric'''
df_salary_copy_numeric['salary_range'] = pd.to_numeric(df_salary_copy_numeric['salary_range'], errors='coerce')
print(df_salary_copy_numeric['salary_range'].value_counts())

#%%

'''what is the average salary of fraudulent jobs?'''
df_salary_copy_numeric[df_salary_copy_numeric['fraudulent'] == 1]['salary_range'].mean()
#%%

'''what is the average salary of non fraudulent jobs?'''
print(df_salary_copy_numeric[df_salary_copy_numeric['fraudulent'] == 0]['salary_range'].mean())

#%%
'''categroies for salary_range based on quartiles'''

df_salary_copy_categories = df_salary_copy_numeric.copy()

df_salary_copy_categories['salary_range'] = pd.qcut(df_salary_copy_categories['salary_range'],
                                                    q=4, labels=['low', 'medium', 'high', 'very high'])


print(df_salary_copy_categories['salary_range'].value_counts())


#%%
'''how many percent of all categories are fraudulent'''
print(df_salary_copy_categories['salary_range'].value_counts())
'''show percentage of fraudulent jobs in each salary_range category'''
print(df_salary_copy_categories.groupby('salary_range')['fraudulent'].sum() /
      df_salary_copy_categories.groupby('salary_range')['fraudulent'].count())

'''merge df_salary_copy_categories with df_salary_copy_date with all rows and columns'''

df_categorie_date = pd.merge(df_salary_copy_date, df_salary_copy_categories, how='outer', on=['job_id', 'title', 'location', 'department', 'salary_range', 'company_profile', 'description', 'requirements',
                                                                                     'benefits', 'telecommuting', 'has_company_logo',
                                                                                     'has_questions', 'employment_type', 'required_experience', 'required_education', 'industry', 'function', 'fraudulent'])

'''create cake plot with salary_range categories'''
df_categorie_date['salary_range'].value_counts().plot(kind='pie', autopct='%1.0f%%', figsize=(10, 10))
plt.title('Salary Range Categories')
plt.show()

'''merge df_category_date with df_no_salary_range with all rows and columns'''
df_merge = pd.merge(df_categorie_date, df_no_salary, how='outer', on=['job_id', 'title', 'location', 'department', 'salary_range',
                                                                            'company_profile', 'description', 'requirements',
                                                                                        'benefits', 'telecommuting', 'has_company_logo',
                                                                                        'has_questions', 'employment_type', 'required_experience',
                                                                            'required_education', 'industry', 'function', 'fraudulent'])



'''create plot with df_merge about salary_range'''
df_merge['salary_range'].value_counts().plot(kind='pie', autopct='%1.0f%%', figsize=(10, 10))
plt.title('Salary Range Categories')
plt.show()
