#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','to_messages','bonus','restricted_stock_deferred' \
#,'other','total_payments','from_poi_to_this_person',\
#'shared_receipt_with_poi','salary','from_this_person_to_poi','total_value_ratio'\
]



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
print data_dict
print len(data_dict.keys())
#remove outliers
from sklearn import linear_model
from sklearn.preprocessing import Imputer
countNaN = 0
found = False
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
#print imp.transform(numpy.array(data_dict))
totalVal =0
maxVal = -9999
minVal=9999999
del data_dict['TOTAL']
for val in data_dict.keys():
    person = data_dict[val]
    countNaN =0
    #print person
    person['total_value'] ='NaN'
    person['to_and_from_poi'] = 'NaN'
    if person['total_payments']!= 'NaN':
        person['total_value']=person['total_payments']
    if person['long_term_incentive'] != 'NaN':
        person['total_value'] = float(person['total_value'])+float(person['long_term_incentive'])
    if person['from_this_person_to_poi'] != 'NaN':
        person['to_and_from_poi'] = person['from_this_person_to_poi']
    if person['from_poi_to_this_person'] != 'NaN':
        person['to_and_from_poi'] = float(person['from_this_person_to_poi']) + float(person['from_poi_to_this_person'])
        
    for item  in person:
        if person[item] == 'NaN'  :
            person[item] = 0 
            countNaN=countNaN+1
      
    #remove the outliers which have more than 70% of NaN values
    if countNaN > 18 :
        print 'Deleting Item {0}',val
        del data_dict[val]
        
#deleting the TOTAL as this is an outlier.
del data_dict['BELFER ROBERT']
del data_dict['BHATNAGAR SANJAY']



for val in data_dict.keys():
    person = data_dict[val]
    countNaN =0
    #print person
    person['total_value_ratio'] =0
    if person['bonus']!= 0:
        person['total_value_ratio']=float(person['total_value_ratio'])+float(person['bonus'])
    if person['total_payments']!= 0:
        person['total_value_ratio']=float(person['total_value_ratio'])+float(person['total_payments'])
  
    if person['total_stock_value'] != 0:
        person['total_value_ratio'] = float(person['total_value_ratio'])+float(person['total_stock_value'])
    if person['salary']!= 0:
        person['total_value_ratio']=float(person['total_value_ratio'])/float(person['salary'])
    else:
        person['total_value_ratio']=0      
    print val,' ',person['total_value_ratio']



del data_dict['REDMOND BRIAN L']
del data_dict['OVERDYKE JR JERE C']
del data_dict['IZZO LAWRENCE L']
del data_dict['BAXTER JOHN C']
del data_dict['REYNOLDS LAWRENCE']
del data_dict['LAVORATO JOHN J']
del data_dict['PAI LOU L']
del data_dict['GRAY RODNEY']
del data_dict['BANNANTINE JAMES M']


print 'total length'
print len(data_dict.keys())



my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
print features

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
from sklearn import preprocessing
from sklearn import tree
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
('scaling',preprocessing.MinMaxScaler()),
#('pca',PCA(n_components=3)),
 ('decisiontree',tree.DecisionTreeClassifier())
])
print 'scores'
print selectK.scores_
print 'New set of Features'
print len(features_list)
print selectK.get_support()
# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model

from sklearn.ensemble import RandomForestClassifier
#pca = RandomizedPCA(n_components=2, whiten=True).fit(features)
pca = PCA(n_components=2).fit(features)

print 'Features'
#features = pca.transform(features)
#features = pipeline.fit_transform(features)
#print numpy.array(features)

param_grid = {
         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
from sklearn import svm

from sklearn.svm import SVC
#clf = svm.LinearSVC(fit_intercept=True,max_iter=1000,tol=0.01)

#clf = SVC(100.0,kernel="linear",tol=0.01)
clf = pipeline
#clf = tree.DecisionTreeClassifier()
#clf = linear_model.LogisticRegression(C=1e-3)
#clf = RandomForestClassifier(n_estimators=5,min_samples_leaf=2)
parameters = {'kernel':'linear', 'C':[1, 10]}
#svr = svm.SVC()
#clf = GridSearchCV(svr, parameters)
#clf = GridSearchCV(SVC(kernel='linear', class_weight='auto'), param_grid)
#clf = GaussianNB()
print clf

### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split

#cv=StratifiedKFold(labels,20,True,random_state=42)
#Just splits into the two
#cv=KFold(tot,10,True,random_state=42)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=40)


dump_classifier_and_data(clf, my_dataset, features_list)
