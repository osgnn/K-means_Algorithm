#K-means Algorithm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("marka_satis_birimfiyat_norm.csv")

kolon_eksik_deger_toplami = veriler.isnull().sum()
print(kolon_eksik_deger_toplami)

X = veriler.iloc[:,1:].values

from sklearn.cluster import KMeans

sonuclar = []
for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init='k-means++', n_init = 10,random_state= 0)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)


plt.plot(range(1,11),sonuclar)
plt.title('Küme Sayısı Belirlemek için Dirsek Yöntemi')
plt.Xlabel('Küme Sayısı')
plt.show()

kmeans = KMeans (n_clusters = 6, init='k-means++', random_state= 0)
Y_tahmin= kmeans.fit_predict(X)
print(Y_tahmin)  
for i in range(0,181):
    print(veriler.iloc[i,0])
    print(Y_tahmin[i])


plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=75, c='cyan',label = 'Küme 1')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=75, c='blue', label = 'Küme 2')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=75, c='green', label = 'Küme 3')
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=75, c='yellow', label = 'Küme 4')
plt.scatter(X[Y_tahmin==4,0],X[Y_tahmin==4,1],s=75, c='purple', label = 'Küme 5')
plt.scatter(X[Y_tahmin==5,0],X[Y_tahmin==5,1],s=75, c='magenta', label = 'Küme 6')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 150, c = 'red', label = 'Küme Merkezleri')
plt.title('Marka Segmentasyonu')
plt.xlabel('Adet')
plt.ylabel('Birim Fiyat')
plt.legend()
plt.show()

