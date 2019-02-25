from PIL import Image
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.colors as col


#import images in a matrix 100, 99, 83, 46

examples = []
for i in range(0,360,5):
    examples.append(np.asarray(Image.open('D:\Universita\Intelligenza Artificiale e Robotica\Machine Learning\ML_hmw1\coil-100\obj100__'\
                                          + str(i) + '.png')).ravel())

for i in range(0,360,5):
    examples.append(np.asarray(Image.open('D:\Universita\Intelligenza Artificiale e Robotica\Machine Learning\ML_hmw1\coil-100\obj99__'\
                                          + str(i) + '.png')).ravel())

for i in range(0,360,5):
    examples.append(np.asarray(Image.open('D:\Universita\Intelligenza Artificiale e Robotica\Machine Learning\ML_hmw1\coil-100\obj83__'\
                                          + str(i) + '.png')).ravel())

for i in range(0,360,5):
    examples.append(np.asarray(Image.open('D:\Universita\Intelligenza Artificiale e Robotica\Machine Learning\ML_hmw1\coil-100\obj46__'\
                                          + str(i) + '.png')).ravel())
m = np.asmatrix(examples)

X = preprocessing.scale(m.astype(float))

pca = 100 #specify number of components #################################################################################
X_t = PCA(pca).fit_transform(X)

#calcolo varianza
var_tot = len(X[0,:])
var_expl = []
for i in range(len(X_t[0,:])):
    var_expl.append(np.var(X_t[:,i])/var_tot*100)

print var_expl
print sum(var_expl)
plt.plot(var_expl)
plt.ylabel('% of variance explained')
plt.xlabel('n components')
plt.show()




for i in range(pca-2):
    X_t = np.delete(X_t, 0, 1)

Y = [0]*72 + [1]*72 + [2]*72 + [3]*72
#colors = ['red', 'yellow', 'green', 'grey']



X_train, X_test, y_train, y_test = train_test_split(X_t, Y, test_size=0.4, random_state=0)
clf = GaussianNB()
clf.fit(X_train, y_train)
print clf.score(X_test, y_test)

h = 0.1

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X_t[:, 0].min() - 100, X_t[:, 0].max() + 100
y_min, y_max = X_t[:, 1].min() - 100, X_t[:, 1].max() + 100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
cmap_light = col.ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(X_t[:,0], X_t[:,1], c=Y)

plt.show()
