import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 1000)
import warnings
warnings.filterwarnings('ignore')
#read data
origin_data=pd.read_csv('C:\\Users\\tybty\\data_30days.csv')
    #pd.read_csv('C:\\Users\\tybty\\data_readmitted.csv')
target_data=pd.read_csv('C:\\Users\\tybty\\test.csv')
#fillna(0)
origin_data=origin_data.fillna(0)
target_data=target_data.fillna(0)
#split x and y
train_input=origin_data.iloc[:,3:-1]
train_output=origin_data['readmitted']
target_input=target_data.iloc[:,3:-1]

#Standard Scaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_norm=scaler.fit_transform(train_input)
x_target_norm=scaler.fit_transform(target_input)
print("Mean after Standard Scaler：")
print(x_train_norm.mean(axis=0))
print("STD after Standard Scaler")
print(x_train_norm.std(axis=0))

#SMOTE Sampling
from imblearn.over_sampling import SMOTE
from collections import Counter
print('Original dataset shape {}'.format(Counter(train_output)))
smt = SMOTE(random_state=20)
train_input_new, train_output_new = smt.fit_sample(x_train_norm, train_output)
print('New dataset shape {}'.format(Counter(train_output_new)))
#train set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train_input_new,train_output_new , test_size=0.20, random_state=0)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
logreg = LogisticRegression(fit_intercept=True, penalty='l1')
print("Logistic Regression Cross Validation Score: {:.2%}".format(np.mean(cross_val_score(logreg, x_train, y_train, cv=10))))
logreg.fit(x_train, y_train)
print("Logistic Regression Dev Set score: {:.2%}".format(logreg.score(x_test, y_test)))
y_predict = logreg.predict(x_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
print("Logistic Regression Accuracy is {0:.2f}".format(accuracy_score(y_test, y_predict)))
print("Logistic Regression Precision is {0:.2f}".format(precision_score(y_test, y_predict)))
print("Logistic Regression Recall is {0:.2f}".format(recall_score(y_test, y_predict)))
print("Logistic Regression AUC is {0:.2f}".format(roc_auc_score(y_test, y_predict)))

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
dte = DecisionTreeClassifier(max_depth=28, criterion = "entropy", min_samples_split=10)
print("Decision Tree Cross Validation score: {:.2%}".format(np.mean(cross_val_score(dte,x_train, y_train, cv=10))))
dte.fit(x_train, y_train)
print("Decision Tree Dev Set score: {:.2%}".format(dte.score(x_test, y_test)))
y_predict = dte.predict(x_test)
pd.crosstab(pd.Series(y_test, name = 'Actual'), pd.Series(y_predict, name = 'Predict'), margins = True)
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
print("Decision Tree Accuracy is {0:.2f}".format(accuracy_score(y_test, y_predict)))
print("Decision Tree Precision is {0:.2f}".format(precision_score(y_test, y_predict)))
print("Decision Tree Recall is {0:.2f}".format(recall_score(y_test, y_predict)))
print("Decision Tree AUC is {0:.2f}".format(roc_auc_score(y_test, y_predict)))
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import graphviz
from IPython.display import Image
import pydotplus
from sklearn import tree
dot_dt_q2 = tree.export_graphviz(dte, out_file="DecisionTree.dot", feature_names=train_input.columns, max_depth=2, class_names=["No","Readmitted"], filled=True, rounded=True, special_characters=True)
graph_dt_q2 = pydotplus.graph_from_dot_file('DecisionTree.dot')
Image(graph_dt_q2.create_png())
from matplotlib import pyplot as plt
feature_names = train_input.columns
feature_imports = dte.feature_importances_
most_imp_features = pd.DataFrame([f for f in zip(feature_names,feature_imports)], columns=["Feature", "Importance"]).nlargest(10, "Importance")
most_imp_features.sort_values(by="Importance", inplace=True)
plt.figure(figsize=(10,6))
plt.barh(range(len(most_imp_features)), most_imp_features.Importance, align='center', alpha=0.8)
plt.yticks(range(len(most_imp_features)), most_imp_features.Feature, fontsize=14)
plt.xlabel('Importance')
plt.title('Decision Tree Most important features')
plt.show()

#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
dte = RandomForestClassifier(max_depth=28, criterion = "entropy", min_samples_split=10)
print("Random Forest Cross Validation score: {:.2%}".format(np.mean(cross_val_score(dte, x_train, y_train, cv=10))))
dte.fit(x_train, y_train)
print("Random Forest Dev Set score: {:.2%}".format(dte.score(x_test, y_test)))
y_predict = dte.predict(x_test)
print(pd.crosstab(pd.Series(y_test, name = 'Actual'), pd.Series(y_predict, name = 'Predict'), margins = True))
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
print("Accuracy is {0:.2f}".format(accuracy_score(y_test, y_predict)))
print("Precision is {0:.2f}".format(precision_score(y_test, y_predict)))
print("Recall is {0:.2f}".format(recall_score(y_test, y_predict)))
print("AUC is {0:.2f}".format(roc_auc_score(y_test, y_predict)))
from matplotlib import pyplot as plt
feature_names = train_input.columns
feature_imports = dte.feature_importances_
most_imp_features = pd.DataFrame([f for f in zip(feature_names,feature_imports)], columns=["Feature", "Importance"]).nlargest(10, "Importance")
most_imp_features.sort_values(by="Importance", inplace=True)
plt.figure(figsize=(10,6))
plt.barh(range(len(most_imp_features)), most_imp_features.Importance, align='center', alpha=0.8)
plt.yticks(range(len(most_imp_features)), most_imp_features.Feature, fontsize=14)
plt.xlabel('Importance')
plt.title('Random Forest Most important features')
plt.show()
y_target=dte.predict(x_target_norm)
target_data['result']=y_target
target_data[['encounter_id','patient_nbr','result']].to_csv('./result_30days.csv')
    #.to_csv('./result_readmitted.csv')

from sklearn.decomposition import PCA
pca=PCA(n_components=5)
pca.fit(x_train,y_train)
x_train_red=pca.transform(x_train)
x_test_red=pca.transform(x_test)


#SVM
from sklearn import svm
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear')
print("Random Forest Cross Validation score: {:.2%}".format(np.mean(cross_val_score(clf, x_train_red, y_train, cv=10))))
clf.fit( x_train_red, y_train)
print("Random Forest Dev Set score: {:.2%}".format(clf.score(x_test_red, y_test)))
y_predict = clf.predict(x_test_red)
#pd.crosstab(pd.Series(y_test, name = 'Actual'), pd.Series(y_predict, name = 'Predict'), margins = True)
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
print("Accuracy is {0:.2f}".format(accuracy_score(y_test, y_predict)))
print("Precision is {0:.2f}".format(precision_score(y_test, y_predict)))
print("Recall is {0:.2f}".format(recall_score(y_test, y_predict)))
print("AUC is {0:.2f}".format(roc_auc_score(y_test, y_predict)))