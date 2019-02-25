from sklearn import neighbors, datasets, preprocessing
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import math
iris = datasets.load_iris()
X = iris.data
Y = iris.target
X_scaled = []

# standardization
for i in X:
    X_scaled.append(list(preprocessing.scale(i)))

X_t = PCA(2).fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_t, Y, test_size=0.4, random_state=0)

n_neighbors = 3
choice_possweights = 1
possweights_labels  = ['uniform', 'distance', '(gaussian with alpha = 1000)']


def userfunction(arrayofdistances):
    alpha = 100 ############################change alpha here
    for i in range(len(arrayofdistances)):
        arrayofdistances[i] = math.e**(-alpha*(arrayofdistances[i]**2))
    #print (arrayofdistances[:3])
    return arrayofdistances

possweights = ['uniform', 'distance', userfunction]

for i in range(1,10):
    n_neighbors = i
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=possweights[choice_possweights]) ############change weight function here
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print ("%.3f" % score)
'''
h = 0.01

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
cmap_light = col.ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

#print Z[0]
for i in range(len(y_train)):
    if y_train[i] == 0:
        plt.plot(X_train[i, 0], X_train[i, 1], 'ro')
    if y_train[i] == 1:
        plt.plot(X_train[i, 0], X_train[i, 1], 'go')
    if y_train[i] == 2:
        plt.plot(X_train[i, 0], X_train[i, 1], 'bo')
plt.axis([x_min, x_max, y_min, y_max])
plt.title('k = '+str(n_neighbors)+', weights = '+str(possweights_labels[choice_possweights])+', score ='+str(score))
plt.savefig(str(n_neighbors)+'k_KNN.png', bbox_inches='tight')
plt.close()
'''