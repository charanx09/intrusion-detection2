import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Import dataset
dataset = pd.read_csv('data1.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# ==========================
# Label Encoding (if needed)
# ==========================
from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

labelencoder_X = LabelEncoder()
X[:, 8] = labelencoder_X.fit_transform(X[:, 8])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# ==========================
# Missing Values
# ==========================
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:50])
X[:, 1:50] = imputer.transform(X[:, 1:50])

# ==========================
# Train-Test Split
# ==========================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ==========================
# Feature Scaling
# ==========================
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ==========================
# LDA (Dimensionality Reduction)
# ==========================
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# ==========================
# LightGBM Classifier
# ==========================
from lightgbm import LGBMClassifier

classifier = LGBMClassifier(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# ==========================
# Evaluation
# ==========================
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix:\n', cm)

# Correct TP, TN, FP, FN
tp = np.sum((y_test == 1) & (y_pred == 1))
tn = np.sum((y_test == 0) & (y_pred == 0))
fp = np.sum((y_test == 0) & (y_pred == 1))
fn = np.sum((y_test == 1) & (y_pred == 0))

# Safe metrics
tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
ppv = tp / (tp + fp) if (tp + fp) != 0 else 0
npv = tn / (tn + fn) if (tn + fn) != 0 else 0
fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
fnr = fn / (tp + fn) if (tp + fn) != 0 else 0
fdr = fp / (tp + fp) if (tp + fp) != 0 else 0
acc = (tp + tn) / (tp + tn + fp + fn)

# Print results
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