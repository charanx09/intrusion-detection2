import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Importing the dataset
dataset = pd.read_csv('data1.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:50])
X[:, 1:50] = imputer.transform(X[:, 1:50])

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

# ==========================
# ✅ AdaBoost Classifier
# ==========================
from sklearn.ensemble import AdaBoostClassifier

classifier = AdaBoostClassifier(n_estimators=50, random_state=0)
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred))

# ==========================
# Confusion Matrix Metrics
# ==========================

cm = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix:\n', cm)

# Correct TP, TN, FP, FN
tp = np.sum((y_test == 1) & (y_pred == 1))
tn = np.sum((y_test == 0) & (y_pred == 0))
fp = np.sum((y_test == 0) & (y_pred == 1))
fn = np.sum((y_test == 1) & (y_pred == 0))

# Metrics
tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
ppv = tp / (tp + fp) if (tp + fp) != 0 else 0
npv = tn / (tn + fn) if (tn + fn) != 0 else 0
fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
fnr = fn / (tp + fn) if (tp + fn) != 0 else 0
fdr = fp / (tp + fp) if (tp + fp) != 0 else 0
acc = (tp + tn) / (tp + tn + fp + fn)

# Printing results
print('\nTrue Positive : %d' % tp)
print('True Negative : %d' % tn)
print('False Positive : %d' % fp)
print('False Negative : %d' % fn)

print('\nSensitivity (Recall / TPR): %f' % tpr)
print('Specificity (TNR): %f' % tnr)
print('Precision (PPV): %f' % ppv)
print('Negative Predictive Value: %f' % npv)
print('False Positive Rate: %f' % fpr)
print('False Negative Rate: %f' % fnr)
print('False Discovery Rate: %f' % fdr)

print('\nOverall Accuracy: %f' % acc)