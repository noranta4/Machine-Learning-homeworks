from sklearn import neighbors, datasets, preprocessing
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import math
iris = datasets.load_iris()
X = iris.data[:,[0,1]] # delete first two columns
Y = iris.target
X = preprocessing.scale(X)

#split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=4) # split 50% training, 50%test (and validation)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.6, random_state=4) # split test and validation

#KFOLD
kfold_check = 0
if kfold_check != 0:
    X_train = np.concatenate((X_train, X_validation))
    y_train = np.concatenate((y_train, y_validation))
# SVM ############################################################
C_list = [0.1, 1., 10., 100., 1000.]
gamma_list = [10**(-9), 10**(-7), 10**(-5), 10**(-3), 10**(-1), 10]
score_list = []
kernel_list = ['linear', 'rbf']
kernel = kernel_list[1]
#C_list = [1.]
print '\t'.join(map(str, [-1]+C_list))

for gamma in gamma_list:
    for c in C_list:
        clf = SVC(C=c, kernel=kernel, gamma=gamma)
        clf.fit(X_train, y_train)
        score = clf.score(X_validation, y_validation)
        score_list.append(score)

        #print clf.predict([[3,2]]) #rosso 0, verde 1, blu 2

    print '\t'.join(map(str, [gamma] + score_list))
    score_list = []

'''
# score plot
plt.semilogx(C_list, score_list)
plt.axis([0.001, 1000, 0, 1])
plt.ylabel('score')
plt.xlabel('C')
plt.grid(True)
plt.title('Scores with ' + kernel + ' kernel')
plt.show()
'''
# score with test set
best_c = 10
best_gamma = 0.1
print 'The best c value is: ', best_c
print 'The best gamma value is: ', best_gamma
clf = SVC(C=best_c, kernel=kernel, gamma = best_gamma)
clf.fit(X_train, y_train)
score_test = clf.score(X_test, y_test)
print 'The score with the ' + kernel + ' kernel is: ', score_test

#plotting
# Plot the decision boundary
h = 0.01
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
cmap_light = col.ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# Plot data
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, label='training set')
#plt.scatter(X_validation[:,0], X_validation[:,1], c = y_validation, label = 'validation set')
plt.title(kernel + ' kernel, C = ' + str(best_c) + ', gamma = ' + str(best_gamma) + '\nScore = ' + str(score_test))
#plt.legend()
plt.axis([x_min, x_max, y_min, y_max])
plt.show()
