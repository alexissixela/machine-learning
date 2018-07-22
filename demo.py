from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network
from sklearn import gaussian_process

clf = tree.DecisionTreeClassifier()


# CHALLENGE - create 3 more classifiers...
# 1
clf1 = ensemble.AdaBoostClassifier()

# 2
clf2 = neural_network.MLPClassifier()

# 3
clf3 = gaussian_process.GaussianProcessClassifier()


#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)

prediction = clf.predict([[190, 70, 43]])
prediction1 = clf1.predict([[190, 70, 43]])
prediction2 = clf2.predict([[190, 70, 43]])
prediction3 = clf3.predict([[190, 70, 43]])

# CHALLENGE compare their reusults and print the best one!

print(prediction)
print(prediction1)
print(prediction2)
print(prediction3)