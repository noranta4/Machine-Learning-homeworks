from sklearn import datasets, preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score
from sklearn.metrics import confusion_matrix
import numpy as np
import math
digits = datasets.load_digits()
X = digits.data
y = digits.target
X = X[y<5] #get classes 0, 1, 2, 3, 4
y = y[y<5]
X_scaled = []

# standardization
for i in X:
    X_scaled.append(list(preprocessing.scale(i)))

X_t = PCA(2).fit_transform(X_scaled)

# KMEANS #######################################################################

for num_clusters in range(3,5):
    kmeans = KMeans(num_clusters)
    kmeans.fit(X_t)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    h = 0.02
    x_min, x_max = X_t[:, 0].min() - 1, X_t[:, 0].max() + 1
    y_min, y_max = X_t[:, 1].min() - 1, X_t[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Set2)
    # plot data with the correct color
    plt.scatter(X_t[:, 0], X_t[:, 1], c=y)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('K-means\n' + str(num_clusters) +' clusters')
    #plt.savefig(str(num_clusters)+'cl_KM.png', bbox_inches='tight')
    plt.close()

y_predicted = kmeans.predict(X_t)
print normalized_mutual_info_score(y,y_predicted)
############################################################
# GMM ######################################################

homogeneity = []
nmi = []
purity = []

max_num_clusters = 5
for num_clusters in range(2,max_num_clusters):
    gmm = GMM(num_clusters)
    gmm.fit(X_t)

    #'''
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    h = 0.02
    x_min, x_max = X_t[:, 0].min() - 1, X_t[:, 0].max() + 1
    y_min, y_max = X_t[:, 1].min() - 1, X_t[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = gmm.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Set2)
    #'''

    # performaces
    y_predicted = gmm.predict(X_t)
    homogeneity.append(homogeneity_score(y,y_predicted))
    nmi.append(normalized_mutual_info_score(y,y_predicted))
    m = confusion_matrix(y_predicted,y)
    maximums = []
    for row in m:
        maximums.append(row.max())
    purity.append(float(sum(maximums))/len(y))
    #print m, maximums, purity

    # plot data with the correct color
    plt.scatter(X_t[:, 0], X_t[:, 1], c=y)
    plt.title('KGaussian Mixture models\n' + str(num_clusters) +' clusters')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    #plt.savefig(str(num_clusters)+'cl_GMM.png', bbox_inches='tight')
    plt.close()
plt.plot(range(2,max_num_clusters), homogeneity, label='homogeneity')
plt.plot(range(2,max_num_clusters), nmi, label='nmi')
plt.plot(range(2,max_num_clusters), purity, label='purity')
plt.legend(loc=4)
plt.title('Evaluation of GMM clustering')
plt.xlabel('number of clusters')
plt.ylabel('performance')
plt.show()
print normalized_mutual_info_score(y,y_predicted)



