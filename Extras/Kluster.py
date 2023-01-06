import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('Melbourne_housing.csv')

print(df.columns)

'''change all values to numeric'''

df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Suburb'] = pd.to_numeric(df['Suburb'], errors='coerce')
df['Address'] = pd.to_numeric(df['Address'], errors='coerce')
df['Rooms'] = pd.to_numeric(df['Rooms'], errors='coerce')
df['Type'] = pd.to_numeric(df['Type'], errors='coerce')
df['Method'] = pd.to_numeric(df['Method'], errors='coerce')
df['SellerG'] = pd.to_numeric(df['SellerG'], errors='coerce')
df['Date'] = pd.to_numeric(df['Date'], errors='coerce')
df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
df['Week'] = pd.to_numeric(df['Week'], errors='coerce')
df['CPI'] = pd.to_numeric(df['CPI'], errors='coerce')
df['LastCPI'] = pd.to_numeric(df['LastCPI'], errors='coerce')
df['Distance2'] = pd.to_numeric(df['Distance2'], errors='coerce')

print(df.corr())
print(df.isnull().sum())
'''insert missing values'''

# Pipeline 
df['Price'].fillna(df['Price'].mean(), inplace=True)
df['Suburb'].fillna(df['Suburb'].mean(), inplace=True)
df['Address'].fillna(df['Address'].mean(), inplace=True)
df['Rooms'].fillna(df['Rooms'].mean(), inplace=True)
df['Type'].fillna(df['Type'].mean(), inplace=True)
df['Method'].fillna(df['Method'].mean(), inplace=True)
df['SellerG'].fillna(df['SellerG'].mean(), inplace=True)
df['Date'].fillna(df['Date'].mean(), inplace=True)
df['Distance'].fillna(df['Distance'].mean(), inplace=True)
df['Week'].fillna(df['Week'].mean(), inplace=True)
df['CPI'].fillna(df['CPI'].mean(), inplace=True)
df['LastCPI'].fillna(df['LastCPI'].mean(), inplace=True)
df['Distance2'].fillna(df['Distance2'].mean(), inplace=True)

print(df.isnull().sum())

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

'''remove 'type, method, sellerG, date,  '''

df = df.drop(['Type', 'Method', 'SellerG', 'Date', 'Suburb', 'Address'], axis=1)
print(df.isnull().sum())
X = df[['Distance', 'Price']]

X = StandardScaler().fit_transform(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeansmodel = KMeans(n_clusters=3, init='k-means++', random_state=0)
y_kmeans = kmeansmodel.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')

plt.scatter(kmeansmodel.cluster_centers_[:, 0], kmeansmodel.cluster_centers_[:, 1], s=300, c='black',
            label='Centroids')
plt.title('Clusters of houses')
plt.xlabel('Price')
plt.ylabel('Distance')
plt.legend()
plt.show()

'''make pairplot with seaborn'''

import seaborn as sns

sns.pairplot(df, hue='Price')
plt.show()

