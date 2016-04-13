#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy

### Task 1: Select what features you'll use.

features_list = ['poi','age','sex','steroid','antivirals','fatigue','malaise','anorexia','liver_big',\
                 'liver_Firm','spleen_palpable','spiders','ascites','varices','bilirubin',\
                 'phosphate','sgot','albumin','protime','histology']



 
# You will need to use more features

### Load the dictionary containing the dataset
with open("liver.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
print data_dict
print len(data_dict.keys())
#remove outliers
from sklearn import linear_model
from sklearn.preprocessing import Imputer
#ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)
countNaN = 0
found = False
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
#print imp.transform(numpy.array(data_dict))
totalVal =0
maxVal = -9999
minVal=9999999


print 'total length'
print len(data_dict.keys())



my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
#selectK = VarianceThreshold(0.8*(0.2))

print 'features after scaling'
print numpy.array(features)
selectK = SelectKBest(f_classif,len(features_list)-1)

#selectK = SelectKBest(chi2,6)
#selectK = SelectPercentile(f_classif,60)
features = selectK.fit_transform(features,labels)

in_max_scaler = preprocessing.MinMaxScaler()
features = in_max_scaler.fit_transform(features)
from sklearn.pipeline import Pipeline
pipeline = Pipeline(steps=[    
#('scaling',preprocessing.MinMaxScaler()),
#('pca',PCA(n_components=2)),
# ('decisiontree',tree.DecisionTreeClassifier())
('Gaussian',GaussianNB()),
])

print 'scores'
print selectK.scores_
print 'New set of Features'
print len(features_list)
print selectK.get_support()

pca = PCA(n_components=2).fit(features)

param_grid = {
         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
from sklearn import svm
from sklearn import tree
from sklearn.svm import SVC
#clf = svm.LinearSVC(fit_intercept=True,max_iter=1000,tol=0.01)

#clf = SVC(100.0,kernel="linear",tol=0.01)
#clf = tree.DecisionTreeClassifier()
#clf = linear_model.LogisticRegression(C=1e-3)
#clf = RandomForestClassifier(n_estimators=5,min_samples_leaf=2)
parameters = {'kernel':'linear', 'C':[1, 10]}
#svr = svm.SVC()
#clf = GridSearchCV(svr, parameters)
#clf = GridSearchCV(SVC(kernel='linear', class_weight='auto'), param_grid)
#clf = GaussianNB()
clf = pipeline
print clf


dump_classifier_and_data(clf, my_dataset, features_list)