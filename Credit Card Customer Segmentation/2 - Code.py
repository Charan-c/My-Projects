import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
credit_card = pd.read_csv('creditcard.csv')

# Remove duplicate rows
credit_card.drop_duplicates(inplace=True)

# Handle missing values by imputing the median value
credit_card.fillna(credit_card.median(), inplace=True)

# Scale the features
scaler = StandardScaler()
credit_card_scaled = scaler.fit_transform(credit_card.drop(['CUSTID'], axis=1))

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of each feature
credit_card.hist(bins=30, figsize=(20,20))
plt.show()

# Explore the correlation between different features
plt.figure(figsize=(20,10))
sns.heatmap(credit_card.corr(), annot=True, cmap='coolwarm')
plt.show()

# Analyze the demographics of the customers
sns.countplot(x='TENURE', data=credit_card)
plt.title('Customer Tenure')
plt.show()

from sklearn.cluster import KMeans

# Determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(credit_card_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Perform K-means clustering
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(credit_card_scaled)
credit_card['Cluster'] = kmeans.labels_

# Visualize the clusters using scatter plots
sns.scatterplot(x='PURCHASES', y='PAYMENTS', hue='Cluster', data=credit_card)
plt.title('Credit Card Clusters')
plt.show()

# Analyze the spending patterns of the clusters
sns.boxplot(x='Cluster', y='PURCHASES', data=credit_card)
plt.title('Credit Card Purchases by Cluster')
plt.show()

# Analyze the demographics of the clusters
sns.countplot(x='Cluster', hue='TENURE', data=credit_card)
plt.title('Credit Card Customer Segmentation')
plt.show()

# Summarize the insights gained from the clustering analysis
cluster_summary = credit_card.groupby('Cluster').agg({'BALANCE': 'mean', 'PURCHASES': 'mean', 'CREDIT_LIMIT': 'mean', 'TENURE': 'mean'}).reset_index()
print(cluster_summary)

# Provide recommendations to the bank based on the customer segmentation results
# For example, the bank could tailor its marketing efforts to each customer segment based on their spending patterns and demographics.

# Identify any areas for future research or improvement in the customer segmentation process
# For example, the bank could consider using additional data sources or more advanced clustering algorithms to improve the accuracy of the segmentation results.
