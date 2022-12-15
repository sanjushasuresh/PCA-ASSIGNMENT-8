# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:21:48 2022

@author: SANJUSHA
"""

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("wine.csv")
df
df.shape
# Droping the first column as per the question
df=df.drop("Type",axis=1) 
df
# Type is a column where clustering is already done.
df.shape
df.info()
df.duplicated()
df[df.duplicated()]

# Standardization
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
df=SS.fit_transform(df)
df=pd.DataFrame(df)

# PCA
from sklearn.decomposition import PCA
pca=PCA()
Y=pca.fit_transform(df)
percentage=pca.explained_variance_ratio_

df_new=pd.DataFrame(data=Y,columns=['P0C1','P0C2','P0C3','P0C4','P0C5','P0C6','P0C7','P0C8','P0C9','P0C10','P1C1','P1C2','P1C3'])
df_new
df_new.shape

# After PCA we are taking first three columns because the more percentage of data is present in first three columns
X=df_new.iloc[:,0:3]
X
X.shape


### K-Means clustering ###
# To check how many clusters are required
from sklearn.cluster import KMeans
inertia=[]
for i in range(1,10):
    km=KMeans(n_clusters=i,random_state=10)
    km.fit(X)
    inertia.append(km.inertia_)
print(inertia)

# Elbow method to see variance in inertia by clusters
plt.plot(range(1,10),inertia)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("inertia")
plt.show()
# From the graph we can see that the optimal number of clusters is 3

# Scree plot
import seaborn as sns
d1 = {"kvalue": range(1, 10),'inertiavalues':inertia}
d2 = pd.DataFrame(d1)
sns.barplot(x='kvalue',y="inertiavalues", data=d2) # kvalue=clusters
# Here the variance in inertia b/w 3rd and 4th cluster is less so we can go with 3 clusters

KM=KMeans(n_clusters=3,n_init=30)
Y=KM.fit_predict(X)
Y=pd.DataFrame(Y)
Y.value_counts()
df1=pd.concat([X,Y],axis=1)


### Hierarchical clustering ###
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch # for creating dendrogram 
z = linkage(X, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Features')
plt.ylabel('Wine')
sch.dendrogram(z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,)  # font size for the x axis labels
plt.show()
# The dendrogram shows 4 main clusterings, and i chose to draw the line b/w 6 and 8.

from sklearn.cluster import AgglomerativeClustering
AC=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='complete')
Y=AC.fit_predict(X)
Y=pd.DataFrame(Y)
Y.value_counts()
df2=pd.concat([X,Y],axis=1)

# Inference : The no. of clusters obtained using K-Means is 3 which is the same number of 
# clusters with the original data. Whereas, using hierarchical clustering we obtained 4 clusters.
