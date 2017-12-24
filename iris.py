from scipy.spatial import distance
def euc(a,b):
    return distance.euclidean(a,b)
import random
class KNNscrappy():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        prediction = []
        for row in x_test:
            label = self.closest(row)
            prediction.append(label)

        return prediction

    def closest(self,row):
        best_dist = euc(row, x_train[0])
        index = 0
        for i in range(1,len(self.x_train)):
            dist = euc(row,self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                index = i
        return self.y_train[index]

import numpy as np
from sklearn import datasets
#from sklearn import tree
iris = datasets.load_iris()
x = iris.data
y = iris.target
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = .5)

#from sklearn.neighbors import KNNscrappy
clf = KNNscrappy()

clf.fit(x_train, y_train)

prediction = clf.predict(x_test)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, prediction))

#iris = load_iris()
#print (iris.feature_names)
#print (iris.target_names)

#training data
#test_id = [0,50,100]
#train_target = np.delete(iris.target,test_id)
#train_data = np.delete(iris.data,test_id,axis = 0)

#test_target = iris.target[test_id]
#test_data = iris.data[test_id]

#clf = tree.DecisionTreeClassifier()
#clf.fit(train_data,train_target)
