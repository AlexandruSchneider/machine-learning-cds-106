import re
import pandas as pd

def process_location(df, target_df):
    # around 2% missing values --> drop them
    df = df.dropna(subset=['location'])
    location_split = list(df['location'].str.split(',').values)
    for i, location in enumerate(location_split):
        for j, loc in enumerate(location):
            # strip whitespaces
            location_split[i][j] = loc.strip()
            # replace empty strings with 'Unspecified'
            if loc == '':
                location[j] = 'Unspecified'
    location_split = less_then_3_values(location_split)
    location_split = exactly_3_values(location_split)
    location_split = more_then_3_values(location_split)
    return new_dataframe(location_split, df, target_df)


def less_then_3_values(location_split):
    for i, location in enumerate(location_split):
        if location == ['']:
            location_split[i] = ['Unspecified']*3
        if len(location) == 1:
            location_split[i] += ['Unspecified']*2
        if len(location) == 2:
            location_split[i] += ['Unspecified']
    return location_split


def exactly_3_values(location_split):
    for i, location in enumerate(location_split):
        if location[1] == '':
            location_split[i][1] = 'Unspecified'
    return location_split

    
def more_then_3_values(location_split):
    countries = ['Israel', 'Sweden', 'Greece', 'Philippines', 'D.C.', 'California']
    for i, location in enumerate(location_split):
        # regex checks if state or country is in 4th row but only starting not in the middle of the string
        if len(location) > 3 and (re.search(f'^{location[1]}', location[3]) or 
                                  re.search(f'^{location[0]}', location[3]) or 
                                  location[-1] == 'Unspecified' or
                                  location[-1] == '' or
                                  location[-1] == ' ' or
                                  location[3] in countries):
            location_split[i] = location[:3]
    
    for i, location in enumerate(location_split):
        if len(location) > 3 and (location[1] == 'Unspecified' and location[3].isupper() and len(location[3]) == 2):
            location_split[i] = location[0] + location[3] + location[2]
    
    for i, location in enumerate(location_split):
        if type(location) == str:
            country = location[:2]
            state, city = get_state_and_city_from_string(location)
            location_split[i] = [country, state, city]
            
    return location_split


def get_state_and_city_from_string(location):
    state = ''
    safe_state_index = 0
    for i in range(1, len(location)):
        if location[i].isupper():
            state += location[i]
            safe_state_index = i
    city = location[safe_state_index:]
    cities = []
    city_holding = ''
    for i, letter in enumerate(city):
        if i == 0:
            city_holding += letter
        elif not letter.isupper():
            city_holding += letter
        else:
            cities.append(city_holding)
            city_holding = letter
    # last character is start of city name
    return state[:-1], cities
            

def new_dataframe(location_split, dff, target_df):
    df = dff.copy()
    # check length of location_split and df.location
    df = df.assign(location=location_split)
    liste_df = df.values.tolist()
    new_rows = []
    fradulent = []
    for i, row in enumerate(liste_df):
        for j, element in enumerate(row):
            # check for location
            if j == 1:
                location = element
                if len(location) > 3:
                    # add new rows with the additional cities
                    new_row = row.copy()
                    # get index of row in target_df
                    index = liste_df.index(row)
                    liste_df[i][1] = location[:3]
                    for k in range(3, len(location)):
                        new_row[1] = [location[0], location[1]]
                        new_row[1] += [location[k]]
                        new_rows.append(new_row.copy())
                        # add the job fraudulency
                        fradulent.append(target_df.iloc[index])
    liste_df += new_rows 
    # append fraudulent to target_df
    target_df = pd.concat([target_df, pd.Series(fradulent)])
    
    new_liste = []
    for i, row in enumerate(liste_df):
            location = row[1]
            row = row[:1] + location + row[2:]
            new_liste.append(row)
            
            
            
    cols = df.columns.tolist()
    cols.remove('location')
    cols.insert(1, 'country')
    cols.insert(2, 'state')
    cols.insert(3, 'city')
    
    
    # check if all rows have 3 strings
    for i, row in enumerate(new_liste):
        for j in range(1, 4):
            if type(row[j]) != str:
                row[j] = 'Unspecified'
    
    df_new = pd.DataFrame(new_liste, columns=cols)
    df_new.fillna('Unspecified', inplace=True)
    # dtype to category
    df_new = df_new.astype({'country': str, 'state': str, 'city': str})
    print('Preprocessing location DONE!')
    return df_new, target_df



