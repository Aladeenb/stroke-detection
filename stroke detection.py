from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Data pre-processing

# Hot Encoding, converting categorical attributes into numerical

# Drop id column and delete rows with missing values (from bmi column)
data = pd.read_csv('healthcare-dataset-stroke-data.csv').drop(['id'], axis = 1).dropna()

# 2) gender: "Male", "Female" or "Other"
data = data.replace(["Male", "Female", "Other"] , [0, 1, 2])

# 6) ever_married: "No" or "Yes"
data = data.replace(["Yes", "No"] , [1, 2])

# 7) work_type: "children", "Govt_job", "Never_worked", "Private" or "Self-employed"
data = data.replace(["children", "Govt_job", "Never_worked", "Private", "Self-employed"] ,
                [0, 1, 2, 3, 4])

# 8) Residence_type: "Rural" or "Urban"
data = data.replace(["Rural", "Urban"] , [1, 2])

# 11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"
data = data.replace(["formerly smoked", "never smoked", "smokes", "Unknown"] ,
                [0, 1, 2, 3])

# Allocate inputs
X = data.iloc[:, :10] # except column 10 (stroke result)

# Allocate outputs
y = data.iloc[:, 10]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# ========================================================================================
# Build model using Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

lr = LogisticRegression(random_state = 1, max_iter=400).fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Classification reports, confusion matrix plot and accuracy
print("==================================================================================")
print("LOGISTIC REGRESSION\n\n",
      "classification report\n", 
      classification_report(y_test, y_pred_lr),
      "\nconfusion matrix\n",
      confusion_matrix(y_test, y_pred_lr),
      "\n\nAccuracy =",
      accuracy_score(y_test, y_pred_lr)
      )
cm = confusion_matrix(y_test, y_pred_lr) 
display = ConfusionMatrixDisplay(
    confusion_matrix = cm, display_labels = lr.classes_)
display.plot()
plt.show()
# print("y = ", y_pred_lr)
# print("----------------------------------------------------------------------")

# ========================================================================================
# Build model using SVM (Support Virtual Machine)

from sklearn import svm

d = svm.SVC()

d.fit(X_train, y_train)
y_pred_svm = d.predict(X_test)
# Classification reports, confusion matrix plot and accuracy
print("==================================================================================")
print("SVM (Support Virtual Machine)\n\n",
      "classification report\n", 
      classification_report(y_test, y_pred_svm),
      "\nconfusion matrix\n",
      confusion_matrix(y_test, y_pred_svm),
      "\n\nAccuracy =",
      accuracy_score(y_test, y_pred_svm)
      )
cm = confusion_matrix(y_test, y_pred_svm) 
display = ConfusionMatrixDisplay(
    confusion_matrix = cm, display_labels=d.classes_)
display.plot()
plt.show()

# =======================================================================================
# Build model using K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors = 6)
neigh.fit(X_train, y_train)
y_pred_neigh = neigh.predict(X_test)

# Classification reports, confusion matrix plot and accuracy
print("==================================================================================")
print("K-NEAREST NEIGHBORS\n\n",
      "classification report\n", 
      classification_report(y_test, y_pred_neigh),
      "\nconfusion matrix\n",
      confusion_matrix(y_test, y_pred_neigh),
      "\n\nAccuracy =",
      accuracy_score(y_test, y_pred_neigh)
      )
# print("y = ", y_pred_neigh)
cm = confusion_matrix(y_test, y_pred_neigh) 
display = ConfusionMatrixDisplay(
    confusion_matrix = cm, display_labels = neigh.classes_)
display.plot()
plt.show()


