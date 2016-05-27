import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

'''Part 1: Acquire the Data'''
# SQL command to query database and save the results as a csv
'''
\f ','              sets comma to separate results
\a                  sets output to be unaligned
\t                  sets it to show only rows (tuples)
\o titanic.csv      sets output file to titanic.csv
SELECT * FROM train;
'''
# Explanation of some of the columns:
'''
pClass: passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
sibSp: number of siblings/spouses aboard
parCh: number of parents/children aboard
ticket: ticket number
embarked: port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
'''


'''Part 2: Exploratory Data Analysis'''
# Set the column names and load the csv into a DataFrame
cols = ['passengerID','survived','pClass','last_name','first_name','sex','age',
        'sibSp','parCh','ticket','fare','cabin','embarked']
df = pd.read_csv('titanic.csv',names=cols)

# Check the head and data types
print 'Exploratory Data Analysis:'
print df.head()
print df.dtypes
# Check the null counts
print df.isnull().sum(), '\n\n'

'''Part 3: Data Wrangling'''
# We see cabin is missing 75% of values, so we will drop this column
df.drop('cabin', axis=1, inplace=True)

# Age is missing 708 values. I experimented with replacing missing values with
# the mean age, and also dropping all rows that were missing the age value.
# The models performed better when rows were dropped instead of replaced with
# the mean, so we will go with that method.
# df.age = df.age.replace(np.nan,df.age.mean())

# Map the sex column
df.sex = df.sex.map({'male':'0','female':'1'})
df.sex = pd.to_numeric(df.sex,errors='coerce')
# Map the embarked column
df.embarked = df.embarked.map({'C':0,'Q':1,'S':2})

# Drop rows with null values
df.dropna(axis=0,how='any',inplace=True)

'''Part 4: Logistic Regression and Model Validation'''
# Select the target and feature columns
target_cols = ['pClass','sex','age','sibSp','parCh','fare','embarked']
X = df[target_cols]
y = df['survived']

# A simple logistic regression on X and y
log = LogisticRegression()
log.fit(X,y)
predictions = log.predict(X)
acc = metrics.accuracy_score(y,predictions)
print 'Simple logistic regression accuracy: %f' % acc
# Check the coefficients to see correlations
print 'Logistic regression coefficients:\n%s' % str(log.coef_)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2)
# Fit the logistic model on the training set
log.fit(X_train,y_train)
# Use the model to predict the classification of the testing set
predicted = log.predict(X_test)
predicted_prob = log.predict_proba(X_test)
acc = metrics.accuracy_score(y_test,predicted)
print 'The first 5 prediction probabilites for our test set:\n%s' % str(predicted_prob[0:5])
print 'Logistic regression accuracy (of test set): %f' % acc

# Use cross-validation on the testing set
cvscores = cross_val_score(log,X_test,y_test,cv=12,n_jobs=1)
print 'Logistic regression accuracy using cross-val (of test set): %f' % cvscores.mean()

# Check the classification report
print 'Classification report for logistic regression:'
print metrics.classification_report(y_test,predicted)
# Check the confusion matrix
print 'Confusion matrix for logistic regression:'
print metrics.confusion_matrix(y_test,predicted), '\n'
# We see that we have a F1 score of .84, which is a pretty good score and also
# closely matches our accuracy score of .83.

# Plot the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted_prob[:,1])
auc = metrics.auc(fpr,tpr)
plt.plot(fpr, tpr, label='AUC = %f' % auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve for Logistic Regression')
plt.legend(loc='lower right')
plt.show()
# The dotted red line represents the worst possible model. It shows that for
# every increase in TP, we are also getting a FP.

# The ROC curve shows the relationship between our TPR and FPR.
# We see that for an increase in our TRP we are also introducing more FPs to our
# classifications. We want to choose the point along our curve that maximizes
# TPs while also minimizing FPs


'''Part 5: GridSearchCV'''
# Perform a grid search to find the optimal logistic regression parameters
logreg_parameters = {'penalty':['l1','l2'],'C':np.logspace(-5,1,50),
                    'solver':['liblinear']}
grid = GridSearchCV(LogisticRegression(),logreg_parameters,cv=5)
grid.fit(X_train,y_train)
print 'The best logistic regression paramaters are: %s' % str(grid.best_params_)
print 'These parameters give a score of: %f' % grid.best_score_ , '\n'
# This score is almost exactly the same as the vanilla logistic regression

'''Explain the difference between L1 and L2 penalties'''
# Lasso (L1) penalties use the sum of the absolute values of the coefficients.
# Lasso uses both parameter shrinkage and variable selection (when there are
# colinear variables)

# Ridge (L2) penalties use the sum of the squares of the coefficients.
# Ridge only allows us to zero out all or none of the coefficients of the
# variables
'''What hypothetical situations are the ridge and lasso penalties useful?'''
# We would use Lasso if we have a lot of variables and we suspect that some of
# variables are colinear or only some of the variables are contributing to our
# predictions

# We would use Ridge if we think all of our variables have an impact on our
# model


'''Part 6: GridSearch and kNN'''
# Perform a grid search to find the optimal kNN parameters
knn_parameters = {'n_neighbors':range(1,101),'weights':['uniform','distance']}
grid = GridSearchCV(KNeighborsClassifier(),knn_parameters,cv=5)
grid.fit(X_train,y_train)
print 'The best kNN paramaters are: %s' % grid.best_params_
print 'These parameters give a score of: %f' % grid.best_score_

# As we increase our value of k, we are decreasing our variance and increasing
# bias.
# This happens because as we increase k, we are fitting our model more to our
# training data which increases the bias.

# Logistic regression is better than kNN when our training set is small.
# (for large data sets, kNN will perform better but be more computationally
# expensive)
# Logistic regression is also better if we suspect our features are roughly
# linear.

# Fit a new kNN model with the optimal parameters found in GridSearch
kmodel = KNeighborsClassifier(n_neighbors=22,weights='uniform')
kmodel.fit(X_train,y_train)
predicted = kmodel.predict(X_test)
predicted_prob_k = kmodel.predict_proba(X_test)
print 'Confusion matrix for kNN:'
print metrics.confusion_matrix(y_test,predicted), '\n'
# Compared to the confusion matrix for logistic regression, we see that kNN
# performed worse on both TP and TN.

# Plot the ROC curves for the optimized logistic regression model and optimized
# kNN model on the same plot
fpr_k, tpr_k, thresholds_k = metrics.roc_curve(y_test, predicted_prob_k[:,1])
auc_k = metrics.auc(fpr_k,tpr_k)
plt.plot(fpr, tpr, 'b', label='Logistic AUC = %f' % auc)
plt.plot(fpr_k, tpr_k, 'g', label='kNN AUC = %f' % auc_k)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curves for Logistic Regression and kNN')
plt.legend(loc='lower right')
plt.show()


'''Part 7: Precision-Recall (Bonus)'''
# Grid search the same parameters for logistic regression but change the scoring
# function to average_precision
grid = GridSearchCV(LogisticRegression(),logreg_parameters,cv=5,
                                                    scoring='average_precision')
grid.fit(X_train,y_train)
print "The best parameters for logistic regression using scoring='average_precision': %s" % str(grid.best_params_)
print 'These parameters give a score of: %f' % grid.best_score_
# We see that our model now uses a penalty of L1 instead of L2.
# Our C value also increased, which means we are using a weaker regularization
# strength.
predicted = grid.predict(X_test)
predicted_probs = grid.predict_proba(X_test)
print 'Confusion matrix for logistic regression optimized for precision-recall:'
print metrics.confusion_matrix(y_test,predicted) ,'\n'
# Compared to when we optimized for accuracy, we see our new confusion matrix is
# very similar but performs slightly better.
precision, recall, thresholds = metrics.precision_recall_curve(y_test,
                                                        predicted_probs[:,1])
plt.plot(recall,precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0,1.05])
plt.xlim([0.0,1.0])
plt.title('Precision-Recall Curve')
plt.show()


'''Part 8: Decision Trees, Ensembles, Bagging (Bonus)'''
# GridSearch a decision tree classifier model on the data, searching for optimal
# depth.
dtc_params = {'max_depth':range(1,10)}
grid = GridSearchCV(DecisionTreeClassifier(),dtc_params,cv=5,n_jobs=-1)
grid.fit(X_train,y_train)
print 'The best parameters for a decision tree classifier: %s' % str(grid.best_params_)
print 'These parameters give a score of: %f' % grid.best_score_
predicted_prob_d = grid.predict_proba(X_test)
# Create a new decision tree model with the optimal parameters
dtc = DecisionTreeClassifier(max_depth=4)

# Plot all three optimized models ROC curves on the same plot
fpr_d, tpr_d, thresholds_d = metrics.roc_curve(y_test, predicted_prob_d[:,1])
auc_d = metrics.auc(fpr_d,tpr_d)
plt.plot(fpr, tpr, 'b', label='Logistic AUC = %f' % auc)
plt.plot(fpr_k, tpr_k, 'g', label='kNN AUC = %f' % auc_k)
plt.plot(fpr_d, tpr_d, 'y', label='DTC AUC = %F' % auc_d)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curves for Logistic Regression, kNN, and DTC')
plt.legend(loc='lower right')
plt.show()

# Use a Bagging Classifier with the base estimator of your optimal DTC model
bag = BaggingClassifier(dtc)
print 'Decision tree (with bagging) accuracy: %f' % cross_val_score(bag,X_test,
                                                y_test,cv=5,n_jobs=-1).mean()
# We see our accuracy decreased when using bagging compared to a single
# decision tree classifier

# GridSearch the optimal n_estimators, max_samples, and max_features
dtc_params = {'n_estimators':[2,5,10,15,20,30,40,50,75,100],
            'max_samples':[.5,.6,.7,.8,.9,1],
            'max_features':[.5,.6,.7,.8,.9,1]}
grid = GridSearchCV(bag,dtc_params,cv=5,n_jobs=-1)
grid.fit(X_test,y_test)
print 'The best parameters for the bagging classifier: %s' % str(grid.best_params_)

# Create a bagging classifier model with the optimal parameters
bag_best = BaggingClassifier(dtc,max_features=.5,max_samples=.7,n_estimators=15)
bag_best.fit(X_test,y_test)
predicted = bag_best.predict(X_test)
acc = cross_val_score(bag_best,X_test,y_test,cv=5,n_jobs=-1).mean()
print 'These parameters give a score of: %f' % acc
# We see that we get our best performance using an optimized DTC with bagging
