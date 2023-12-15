#!/usr/bin/env python
# coding: utf-8

# ## PCA using RANDOMFOREST

# In[3]:


import pandas as pd
d=pd.read_csv("pd_speech_features.csv")
x=d.drop(columns=['class','id'])
y=d['class']


# In[4]:


from sklearn.preprocessing import MinMaxScaler
mx=MinMaxScaler()
p=mx.fit_transform(x)


# In[5]:


from sklearn.decomposition import PCA
pca=PCA(n_components=50)
X_pca=pca.fit_transform(p)


# In[6]:


import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
n_splits = 27
smote = SMOTE()
# Initialize the StratifiedKFold
skf = StratifiedKFold(n_splits=n_splits)

# Initialize an empty list to store the accuracies of each fold
accuracies = []
conf_matrices = []
precisions = []
recalls = []
f1_scores = []
X=np.array(X_pca)
Y=np.array(y)


# In[8]:


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
for train_index, test_index in skf.split(X, Y):
    # Split the training data further into training fold and validation fold
    X_train_fold, X_val_fold = X[train_index], X[test_index]
    y_train_fold, y_val_fold = Y[train_index], Y[test_index]
    
    # Initialize and train the Random Forest classifier on the training data
    X_train1,Y_train1=smote.fit_resample(X_train_fold,y_train_fold)
    rf_classifier = RandomForestClassifier(    n_estimators=100,  # Number of trees in the forest
    criterion='gini',  # Split quality criterion ('gini' or 'entropy')
    max_depth=None,    # Maximum depth of the tree
    min_samples_split=2,  # Minimum samples required to split an internal node
    min_samples_leaf=1,   # Minimum samples required to be at a leaf node
    max_features='sqrt',  # Number of features to consider when looking for the best split
    bootstrap=True,       # Whether to bootstrap samples when building trees
    random_state=42,      # Seed for random number generator
    n_jobs=-1             )
    rf_classifier.fit(X_train1,Y_train1)

    # Make predictions on the test data
    y_preds = rf_classifier.predict(X_val_fold)

    # Calculate accuracy and store it in the list
    accuracy = accuracy_score(y_val_fold, y_preds) 
    conf_matrix = confusion_matrix(y_val_fold,y_preds)
    precision = precision_score(y_val_fold, y_preds)
    recall = recall_score(y_val_fold, y_preds)
    f1 = f1_score(y_val_fold, y_preds)

    # Append metrics to lists
    accuracies.append(accuracy)
    conf_matrices.append(conf_matrix)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Calculate mean values across folds
mean_accuracy = np.mean(accuracies, axis=0)
mean_conf_matrix = np.mean(conf_matrices, axis=0)
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)
mean_f1_score = np.mean(f1_scores)



print("Mean Accuracy:", mean_accuracy)
print("Mean Confusion Matrix:\n", mean_conf_matrix)
print("Mean Precision:", mean_precision)
print("Mean Recall:", mean_recall)
print("Mean F1 Score:", mean_f1_score)
print(ConfusionMatrixDisplay.from_predictions(y_true = y_val_fold, y_pred = y_preds))


# In[ ]:




