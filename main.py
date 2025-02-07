
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_score, recall_score

# Load the dataset
dataset_path = '/content/letterDataset.csv'
dataset = pd.read_csv(dataset_path)
dataset.columns = dataset.columns.str.replace("'", "").str.strip()

"""


# Checking the data set

"""

# subset of features to simplify visualizations
selected_features = ['x-box', 'y-box', 'width', 'high', 'onpix']

# Ensure selected features are numeric
dataset[selected_features] = dataset[selected_features].apply(pd.to_numeric, errors='coerce')

# Correlation Heatmap to show the most correlated with the class
plt.figure(figsize=(8, 6))
sns.heatmap(dataset[selected_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (Selected Features)")
plt.show()

# Boxplots for or the top 3 features showing the most variance
for col in selected_features[:3]:  # Limiting to first 3 features
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=dataset[col].dropna())
    plt.title(f"Boxplot of {col}")
    plt.show()

# Class Distribution bar plot to visualize the distribution of the class
plt.figure(figsize=(8, 6))
sns.countplot(x='class', data=dataset, palette="pastel", order=dataset['class'].value_counts().index)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

# KDE Plots for Key Features
plt.figure(figsize=(10, 6))
for col in selected_features[:3]:  # 3 features only for somplicity
    sns.kdeplot(dataset[col].dropna(), label=col, fill=True)
plt.title("Feature Density Distributions (Selected Features)")
plt.legend()
plt.show()

# Check for Missing Values
missing_values = dataset.isnull().sum()
print("Missing Values:")
print(missing_values)

# Check for Outliers
# Select numerical columns
numerical_cols = dataset.select_dtypes(include=['float64', 'int64']).columns
outlier_summary = {}

for col in numerical_cols:
    q1 = dataset[col].quantile(0.25)
    q3 = dataset[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = ((dataset[col] < lower_bound) | (dataset[col] > upper_bound)).sum()
    outlier_summary[col] = {
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": outlier_count,
    }

print("\nOutliers Summary:")
for col, summary in outlier_summary.items():
    print(f"{col}: {summary}")

# Removing some outliers that showed up


from sklearn.preprocessing import RobustScaler

# Step 1: Capping (Winsorizing) for features with heavy outliers
features_to_cap = ["x-bar", "xegvy", "y-bar"]  # capping them instead of removing the outliers to preserve data integrity (large outliers)

for col in features_to_cap:
    q1 = dataset[col].quantile(0.25)
    q3 = dataset[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # Cap outliers and convert to int
    dataset[col] = dataset[col].clip(lower=lower_bound, upper=upper_bound).astype(int)

# Step 2: Remove outliers for features where outliers are likely noise
features_to_remove_outliers = ["x-box", "width", "x2ybr"]  # remove outliers (small)

for col in features_to_remove_outliers:
    q1 = dataset[col].quantile(0.25)
    q3 = dataset[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # Remove rows with outliers in these features
    dataset = dataset[(dataset[col] >= lower_bound) & (dataset[col] <= upper_bound)]

# Step 3: Ensure all numerical columns remain integers
numerical_cols = dataset.select_dtypes(include=['float64', 'int64']).columns

# Reformate Columns
dataset.columns = [col.replace("'", "").replace(" ", "_") for col in dataset.columns]
print(dataset.columns)
row_count = dataset.shape[0]
print(f"Total number of rows in the cleaned dataset: {row_count}")
# Get the percentage of data for each letter
label_percentages = dataset['class'].value_counts(normalize=True) * 100
label_percentages = label_percentages.round(2)

# Display the percentages
print(label_percentages)

"""# AI algorithems

# K-NN
"""

# K-NN
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

knn_accuracy =[]

X = dataset.iloc[:, :-1]  # All columns except the last as features
y = dataset.iloc[:, -1]   # The last column as the label


# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensionality
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Optimize k
k_values = range(1, 21)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal k: {optimal_k}")

# Train the final model
knn = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance')
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
knn_accuracy.append(accuracy * 100)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Cross-Validation and k optimization
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


k_values = range(1, 21)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal k: {optimal_k}\n")

# Plot cross-validation scores
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Cross-Validation Accuracy for Different k Values')
plt.grid()
plt.show()

#Hyperparameter experimentation

# Experimenting with weights and distance metrics
from sklearn.metrics import accuracy_score

# Experiment with different weights
weights_options = ['uniform', 'distance']
for weight in weights_options:
    knn = KNeighborsClassifier(n_neighbors=optimal_k, weights=weight)
    knn.fit(X_train, y_train)
    weight_accuracy = knn.score(X_test, y_test)
    print(f"Accuracy with {weight} weights: {weight_accuracy * 100:.2f}%")

# Experiment with different distance metrics
distance_metrics = ['minkowski', 'manhattan', 'euclidean']
for metric in distance_metrics:
    knn = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance', metric=metric)
    knn.fit(X_train, y_train)
    metric_accuracy = knn.score(X_test, y_test)
    knn_accuracy.append(metric_accuracy * 100)
    print(f"Accuracy with {metric} distance metric: {metric_accuracy * 100:.2f}%")

from sklearn.metrics import confusion_matrix

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for K-NN')
plt.show()

# Calculate TP, TN, FP, FN for each class
tp = []
tn = []
fp = []
fn = []

# Iterate through each class
for i in range(len(conf_matrix)):
    TP = conf_matrix[i, i]
    FP = conf_matrix[:, i].sum() - TP
    FN = conf_matrix[i, :].sum() - TP
    TN = conf_matrix.sum() - (TP + FP + FN)

    tp.append(TP)
    fp.append(FP)
    fn.append(FN)
    tn.append(TN)

# Display TP, TN, FP, FN for each class
print("Performance Metrics for Each Class:")
for i, label in enumerate(np.unique(y_test)):
    print(f"Class {label}:")
    print(f"  True Positives (TP): {tp[i]}")
    print(f"  True Negatives (TN): {tn[i]}")
    print(f"  False Positives (FP): {fp[i]}")
    print(f"  False Negatives (FN): {fn[i]}")

# precision and reecall

knn_precision = precision_score(y_test,y_pred, average='macro')
knn_recall = recall_score(y_test,y_pred, average='macro')

"""# SVM"""

# SVM

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

svm_accuracy =[]

X = dataset.iloc[:, :-1]  # All columns except the last as features
y = dataset.iloc[:, -1]   # The last column as the label


# # Normalize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize the SVM classifier with an RBF kernel
svm_model = SVC(kernel='rbf', random_state=42)

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
svm_accuracy.append(accuracy * 100)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

"""The three cells ahead are considered as fine tuning"""

# Cross-Validation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(svm_model, X_scaled, y, cv=5)
print(f"Cross-Validation Accuracy: {scores.mean() * 100:.2f}%")

# Studying the Influence of Hyperparameters
from sklearn.metrics import accuracy_score

Cs = [0.1, 1, 10, 100]
gammas = [1, 0.1, 0.01, 0.001]

for C in Cs:
    for gamma in gammas:
        model = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"C={C}, gamma={gamma} --> Accuracy: {acc * 100:.2f}%")

# Train the model with the best parameters based on the statictic above
best_model = SVC(C=10, gamma=0.1, kernel='rbf', random_state=42)
best_model.fit(X_train, y_train)

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
svm_accuracy.append(accuracy * 100)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for SVM')
plt.show()

# Calculate TP, TN, FP, FN for each class
tp = []
tn = []
fp = []
fn = []

# Iterate through each class
for i in range(len(conf_matrix)):
    TP = conf_matrix[i, i]
    FP = conf_matrix[:, i].sum() - TP
    FN = conf_matrix[i, :].sum() - TP
    TN = conf_matrix.sum() - (TP + FP + FN)

    tp.append(TP)
    fp.append(FP)
    fn.append(FN)
    tn.append(TN)

# Display TP, TN, FP, FN for each class
print("Performance Metrics for Each Class:")
for i, label in enumerate(np.unique(y_test)):
    print(f"Class {label}:")
    print(f"  True Positives (TP): {tp[i]}")
    print(f"  True Negatives (TN): {tn[i]}")
    print(f"  False Positives (FP): {fp[i]}")
    print(f"  False Negatives (FN): {fn[i]}")

#explanation of failure cases


misclassified_indices = np.where(y_test != y_pred)[0]
print(f"Number of Misclassified Samples: {len(misclassified_indices)}")

# Analyze a few misclassified samples
for idx in misclassified_indices[:5]:  # Inspect the first 5 misclassified samples
    print(f"Sample Index: {idx}, True Label: {y_test.iloc[idx]}, Predicted Label: {y_pred[idx]}")

# precision and reecall
svm_precision = precision_score(y_test, y_pred, average='macro')
svm_recall = recall_score(y_test, y_pred, average='macro')

# check if we are having overfitting because accuracy is very high

# Training accuracy
train_accuracy = model.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Test accuracy
test_accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Check for overfitting
if train_accuracy - test_accuracy > 0.05:  # threshold: 5% difference
    print("overfitting")
else:
    print("no overfitting")

"""# Deep learning: Dense Neural Networks (DNN)"""

# Deep learning (DNN)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

dnn_accuracy =[]


# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scale features

# Encode string labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert labels to integers

# Convert to one-hot encoding
y_categorical = to_categorical(y_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.3, random_state=42)

# Define the DNN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    Dropout(0.3),  # Regularization to prevent overfitting
    Dense(64, activation='relu'),  # Hidden layer
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
dnn_accuracy.append(test_accuracy * 100)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Cross-Validation

from sklearn.model_selection import StratifiedKFold

# Encode the labels for stratified cross-validation
y_labels = np.argmax(y_categorical, axis=1)

# Define the number of folds
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_accuracies = []

for train_idx, val_idx in kf.split(X_scaled, y_labels):
    X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
    y_train_fold, y_val_fold = y_categorical[train_idx], y_categorical[val_idx]

    # Define the model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_fold.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(y_categorical.shape[1], activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=10, batch_size=32, verbose=0)

    # Evaluate on the validation set
    val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    fold_accuracies.append(val_accuracy)

# Average accuracy across folds
print(f"Cross-Validation Accuracy: {np.mean(fold_accuracies) * 100:.2f}%")

dnn_accuracy.append(np.mean(fold_accuracies) * 100)

# Studying the Influence of Hyperparameters

from tensorflow.keras.optimizers import Adam

learning_rates = [0.01, 0.001, 0.0001]
results = {}

for lr in learning_rates:
    # Define the model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(y_categorical.shape[1], activation='softmax')
    ])

    # Compile with varying learning rate
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=0)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    results[lr] = test_accuracy

# Print results
for lr, acc in results.items():
    print(f"Learning Rate: {lr}, Test Accuracy: {acc * 100:.2f}%")

# Fine Tuning
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scale features

# Encode string labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert labels to integers

# Convert to one-hot encoding
y_categorical = to_categorical(y_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.3, random_state=42)

# Define the DNN model with additional layers
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer with more neurons
    Dropout(0.3),  # Regularization to prevent overfitting
    Dense(128, activation='relu'),  # Additional hidden layer
    Dropout(0.3),
    Dense(64, activation='relu'),  # Another hidden layer
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')  # Output layer
])

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.001)

# Train the model with early stopping and learning rate scheduling
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, lr_scheduler]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
dnn_accuracy.append(test_accuracy * 100)

# Explanation of Failure Cases

# Convert predictions to labels
y_pred = model.predict(X_test)  # Raw predictions
y_pred_labels = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
y_test_labels = np.argmax(y_test, axis=1)  # Convert one-hot encoded test labels to class labels

# Check for misclassified samples
misclassified_indices = np.where(y_test_labels != y_pred_labels)[0]
print(f"Number of Misclassified Samples: {len(misclassified_indices)}")

# Analyze a few misclassified samples
for idx in misclassified_indices[:5]:
    print(f"Sample Index: {idx}, True Label: {y_test_labels[idx]}, Predicted Label: {y_pred_labels[idx]}")

from sklearn.metrics import confusion_matrix


# Generate confusion matrix
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for DNN')
plt.show()

# Visualization

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# precision and reecall

dnn_y_pred_labels = np.argmax(y_pred, axis=1)  # Convert predictions to label indices
dnn_y_test_labels = np.argmax(y_test, axis=1)  # Convert test set to label indices

dnn_precision = precision_score(y_test_labels, dnn_y_pred_labels, average='macro')
dnn_recall = recall_score(y_test_labels, dnn_y_pred_labels, average='macro')

"""


# Model Comparison

"""

# performance metrics
comparison_results = pd.DataFrame({
    'Model': ['K-NN', 'SVM', 'DNN'],
    'Accuracy': [max(knn_accuracy), max(svm_accuracy), max(dnn_accuracy)],
    'Precision': [knn_precision, svm_precision, dnn_precision],
    'Recall': [knn_recall, svm_recall, dnn_recall]
})

print("Model Comparison:")
print(comparison_results)


print("Model Comparison:")
print(comparison_results)

# Visualize the comparison
comparison_results.set_index('Model')['Accuracy'].plot(kind='bar', figsize=(8, 5), width=0.5)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.xlabel('Models')
plt.ylim(80, 100)
plt.grid(axis='y', alpha=0.7)

# Annotate bars
for idx, value in enumerate(comparison_results['Accuracy']):
    plt.text(idx, value + 0.5, f'{value:.2f}%', ha='center', fontsize=10)

plt.show()

# Combined Precision and Recall Plot
plt.figure(figsize=(8, 5))
precision_recall = comparison_results.set_index('Model')[['Precision', 'Recall']]
precision_recall.plot(kind='bar', width=0.5, color=['orange', 'green'], figsize=(8, 5))

plt.title('Model Precision and Recall Comparison (0 to 1)')
plt.ylabel('Performance (0 to 1)')
plt.xlabel('Models')
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.7)

# Annotate bars
for i, column in enumerate(precision_recall.columns):
    for idx, value in enumerate(precision_recall[column]):
        plt.text(idx - 0.2 + (i * 0.4), value + 0.02, f'{value:.2f}', fontsize=10)

plt.legend(['Precision', 'Recall'], loc='lower right')
plt.show()
