#!/usr/bin/env python
# coding: utf-8

# In[96]:


import pandas as pd
d=pd.read_csv("pd_speech_features.csv")
x=d.drop(columns=['class','id'])
y=d['class']


# In[97]:


from sklearn.preprocessing import MinMaxScaler
mx=MinMaxScaler()
p=mx.fit_transform(x)


# In[98]:


from sklearn.decomposition import PCA
pca=PCA(n_components=50)
X_pca=pca.fit_transform(p)


# In[99]:


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


# In[104]:


from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


for train_index, test_index in skf.split(X, Y):
    # Split the training data further into training fold and validation fold
    X_train_fold, X_val_fold = X[train_index], X[test_index]
    y_train_fold, y_val_fold = Y[train_index], Y[test_index]
    
    # Initialize the SVM classifier
    X_train1,Y_train1=smote.fit_resample(X_train_fold,y_train_fold)
    svm = SVC(    C=1.0,           # Regularization parameter. Controls the trade-off between maximizing the margin and minimizing the classification error.
    kernel='poly',    # Kernel function: 'linear', 'poly', 'rbf' (Gaussian), 'sigmoid', etc.
    gamma='scale',   # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. 'scale' uses 1 / (n_features * X.var()) as the default value.
    degree=3,        # Degree of the polynomial kernel function ('poly'). Ignored by other kernels.
    coef0=0.0,       # Independent term in kernel function. Used in 'poly' and 'sigmoid'.
    shrinking=True,  # Whether to use the shrinking heuristic. Can speed up training for large datasets.
    probability=False,  # Whether to enable probability estimates. Use for probability calibration.
    tol=1e-3,        # Tolerance for stopping criterion.
    class_weight=None,  # Weights associated with classes. Useful for unbalanced datasets.
    random_state=None,  # Seed for random number generator.
    verbose=False   )
    
    # Train the SVM classifier
    svm.fit(X_train1, Y_train1)
    
    # Make predictions on the validation fold
    y_preds = svm.predict(X_val_fold)
    
    # Calculate the accuracy of the predictions
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

