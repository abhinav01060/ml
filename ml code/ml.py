import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, matthews_corrcoef, cohen_kappa_score, auc)
from sklearn.model_selection import GridSearchCV
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

data = '/project/dataset.csv'
df = pd.read_csv(data)
df.columns = df.columns.str.strip()

df.isnull().sum()
df.describe()

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Remaining duplicate rows: {df.duplicated().sum()}")

# Downsample majority class
df_majority = df[df['Role'] == 'AI ML Specialist']
df_minority = df[df['Role'] != 'AI ML Specialist']

df_majority_downsampled = resample(df_majority, replace=False, n_samples=540, random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42).reset_index(drop=True)

X = df_balanced.drop(['Role'], axis=1)
y = df_balanced['Role']

# Convert categorical column 'Role' to numerical values for correlation matrix
df_numeric = df_balanced.copy()
df_numeric['Role'] = LabelEncoder().fit_transform(df_numeric['Role'])

# Check feature correlation
corr_matrix = df_numeric.corr()
high_corr_features = [column for column in corr_matrix.columns if any(abs(corr_matrix[column]) > 0.9)]
print("Highly correlated features:", high_corr_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

# Apply SMOTE with limited oversampling
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train SVM with Stratified K-Fold Cross Validation and shuffle
svm = SVC(C=0.1, kernel='rbf', gamma='scale', class_weight='balanced')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(svm, X_train_resampled, y_train_resampled, cv=skf, scoring='accuracy')
print("Cross-validation Accuracy: ", np.mean(scores))

svm.fit(X_train_resampled, y_train_resampled)
y_pred = svm.predict(X_test_scaled)

print("Test Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Hyperparameter tuning with GridSearchCV
param_grid = {'C': [0.01, 0.1, 1], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}
grid_search = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=skf, scoring='accuracy', verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)

print("Best Parameters:", grid_search.best_params_)

best_svm = grid_search.best_estimator_
y_pred_val = best_svm.predict(X_test_scaled)
print("Final Classification Report:\n", classification_report(y_test, y_pred_val))

# Compute additional classification metrics
precision = precision_score(y_test, y_pred_val, average='weighted')
recall = recall_score(y_test, y_pred_val, average='weighted')
f1 = f1_score(y_test, y_pred_val, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall (TPR): {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_val)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Compute False Positive Rate (FPR), True Positive Rate (TPR), and Specificity
TN = np.diag(cm).sum() - np.diag(cm)
FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)

FPR = FP / (FP + TN)
Specificity = TN / (TN + FP)

y_test_binarized = label_binarize(y_test, classes=np.unique(y))
y_pred_prob = best_svm.decision_function(X_test_scaled)
roc_auc = roc_auc_score(y_test_binarized, y_pred_prob, average='weighted', multi_class='ovr')

print(f"False Positive Rate (FPR) per class: {FPR}")
print(f"Specificity per class: {Specificity}")
print(f"ROC AUC Score: {roc_auc:.4f}")

# Compute additional metrics
mcc = matthews_corrcoef(y_test, y_pred_val)
kappa = cohen_kappa_score(y_test, y_pred_val)
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
print(f"Cohen’s Kappa Score: {kappa:.4f}")

# Plot ROC Curve
plt.figure(figsize=(10, 7))
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_binarized.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curve')
plt.legend(loc='lower right')
plt.show()

