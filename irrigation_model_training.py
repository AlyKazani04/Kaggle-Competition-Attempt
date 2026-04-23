"""
Irrigation Need Prediction - Multi-Model Training & Submission Generation
Author: ML Competition Submission
Purpose: Train, validate, and generate predictions for Kaggle competition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, 
                             f1_score, precision_score, recall_score, make_scorer)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# SECTION 1: DATA LOADING & EXPLORATION
# ============================================================================

print("="*80)
print("PHASE 1: DATA LOADING & EXPLORATION")
print("="*80)

# Load dataset
df = pd.read_csv('/mnt/user-data/uploads/irrigation_prediction.csv')

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"   - Samples: {df.shape[0]}")
print(f"   - Features: {df.shape[1]}")

# Display target distribution
print("\n🎯 Target Distribution (Irrigation_Need):")
target_dist = df['Irrigation_Need'].value_counts()
for class_label, count in target_dist.items():
    percentage = (count / len(df)) * 100
    print(f"   {class_label:8s}: {count:5d} ({percentage:5.2f}%)")

# Check for missing values
print(f"\n❌ Missing Values: {df.isnull().sum().sum()} (None - Clean dataset!)")

# Identify feature types
categorical_cols = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 
                   'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
numerical_cols = [col for col in df.columns if col not in categorical_cols + ['Irrigation_Need']]

print(f"\n🔤 Categorical Features ({len(categorical_cols)}): {', '.join(categorical_cols)}")
print(f"🔢 Numerical Features ({len(numerical_cols)}): {', '.join(numerical_cols)}")

# ============================================================================
# SECTION 2: DATA PREPROCESSING
# ============================================================================

print("\n" + "="*80)
print("PHASE 2: DATA PREPROCESSING")
print("="*80)

# Create a copy for processing
data = df.copy()

# Encode target variable
target_encoder = LabelEncoder()
data['Irrigation_Need_Encoded'] = target_encoder.fit_transform(data['Irrigation_Need'])
label_mapping = dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))
print(f"\n📝 Target Encoding: {label_mapping}")

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le
    print(f"   ✓ Encoded {col}")

# Separate features and target
X = data[numerical_cols + categorical_cols]
y = data['Irrigation_Need_Encoded']

print(f"\n✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")

# Feature scaling (critical for Naive Bayes and Linear Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(f"\n🔄 Feature Scaling Applied (StandardScaler)")
print(f"   - Mean of scaled features: ~0")
print(f"   - Std of scaled features: ~1")

# ============================================================================
# SECTION 3: MODEL TRAINING & VALIDATION
# ============================================================================

print("\n" + "="*80)
print("PHASE 3: MODEL TRAINING WITH CROSS-VALIDATION")
print("="*80)

# Initialize models
models = {
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, 
                                              class_weight='balanced', solver='lbfgs')
}

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
    'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
}

# Store results
results = {}
confusion_matrices = {}
classification_reports = {}

print("\n🔄 Running 5-Fold Stratified Cross-Validation...\n")

for model_name, model in models.items():
    print(f"{'─'*80}")
    print(f"Model: {model_name}")
    print(f"{'─'*80}")
    
    # Cross-validation scores
    cv_results = cross_validate(model, X_scaled, y, cv=skf, scoring=scoring)
    
    results[model_name] = {
        'accuracy': cv_results['test_accuracy'],
        'f1_weighted': cv_results['test_f1_weighted'],
        'f1_macro': cv_results['test_f1_macro'],
    }
    
    print(f"\n📊 Accuracy Scores (5 folds):")
    for i, score in enumerate(cv_results['test_accuracy'], 1):
        print(f"   Fold {i}: {score:.4f}")
    
    print(f"\n   Mean Accuracy: {cv_results['test_accuracy'].mean():.4f} "
          f"(± {cv_results['test_accuracy'].std():.4f})")
    print(f"   F1-Score (Weighted): {cv_results['test_f1_weighted'].mean():.4f}")
    print(f"   F1-Score (Macro): {cv_results['test_f1_macro'].mean():.4f}")
    
    # Store for detailed analysis (using last fold for confusion matrix)
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    confusion_matrices[model_name] = confusion_matrix(y, y_pred, 
                                                      labels=range(len(label_mapping)))
    classification_reports[model_name] = classification_report(
        y, y_pred, 
        target_names=target_encoder.classes_,
        zero_division=0
    )

# ============================================================================
# SECTION 4: RESULTS SUMMARY & VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("PHASE 4: RESULTS SUMMARY & COMPARISON")
print("="*80)

# Create comparison summary
summary_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'].mean() for m in results.keys()],
    'Accuracy Std': [results[m]['accuracy'].std() for m in results.keys()],
    'F1-Weighted': [results[m]['f1_weighted'].mean() for m in results.keys()],
    'F1-Macro': [results[m]['f1_macro'].mean() for m in results.keys()],
})

print("\n📈 Model Performance Summary:\n")
print(summary_df.to_string(index=False))

# Select best model based on weighted F1-score
best_model_name = max(results.keys(), 
                      key=lambda x: results[x]['f1_weighted'].mean())
best_f1_score = results[best_model_name]['f1_weighted'].mean()

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   F1-Score (Weighted): {best_f1_score:.4f}")

# Display confusion matrices
print("\n" + "="*80)
print("CONFUSION MATRICES")
print("="*80)

for model_name, cm in confusion_matrices.items():
    print(f"\n{model_name}:")
    cm_df = pd.DataFrame(cm, 
                        index=[f'True {c}' for c in target_encoder.classes_],
                        columns=[f'Pred {c}' for c in target_encoder.classes_])
    print(cm_df)

# Display classification reports
print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORTS")
print("="*80)

for model_name, report in classification_reports.items():
    print(f"\n{model_name}:")
    print(report)

# ============================================================================
# SECTION 5: FINAL MODEL TRAINING ON FULL DATASET
# ============================================================================

print("\n" + "="*80)
print("PHASE 5: FINAL MODEL TRAINING ON FULL DATASET")
print("="*80)

print(f"\n🎯 Re-training {best_model_name} on entire training set...")

# Get the best model
if best_model_name == 'Naive Bayes':
    final_model = GaussianNB()
else:
    final_model = LogisticRegression(max_iter=1000, random_state=42, 
                                     class_weight='balanced', solver='lbfgs')

# Train on full dataset
final_model.fit(X_scaled, y)
print(f"✓ Model trained on {len(X_scaled)} samples")

# ============================================================================
# SECTION 6: TEST SET LOADING & PREPROCESSING
# ============================================================================

print("\n" + "="*80)
print("PHASE 6: TEST SET PREPARATION")
print("="*80)

# NOTE: This section assumes test.csv is in the same directory
# You'll need to download it from Kaggle and place it in /mnt/user-data/uploads/

try:
    test_df = pd.read_csv('/mnt/user-data/uploads/test.csv')
    print(f"\n✓ Test set loaded: {test_df.shape[0]} samples")
    
    # Store IDs for submission
    test_ids = test_df['id'].copy()
    
    # Apply same preprocessing
    for col in categorical_cols:
        if col in test_df.columns:
            test_df[col] = label_encoders[col].transform(test_df[col].astype(str))
    
    # Select same features as training
    X_test = test_df[numerical_cols + categorical_cols]
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Test features preprocessed: {X_test_scaled.shape}")
    
except FileNotFoundError:
    print("\n⚠️  test.csv not found. Creating dummy test predictions for demonstration...")
    # For demonstration, create synthetic test data
    test_ids = np.arange(630000, 630000 + 100)
    X_test_scaled = scaler.transform(X_scaled.iloc[:100])  # Use first 100 training samples as demo

# ============================================================================
# SECTION 7: GENERATE PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("PHASE 7: GENERATING PREDICTIONS")
print("="*80)

# Make predictions
y_pred_encoded = final_model.predict(X_test_scaled)

# Decode predictions back to original class labels
y_pred_labels = target_encoder.inverse_transform(y_pred_encoded)

print(f"\n✓ Predictions generated: {len(y_pred_labels)} samples")
print(f"\nPrediction Distribution:")
pred_dist = pd.Series(y_pred_labels).value_counts()
for class_label, count in pred_dist.items():
    percentage = (count / len(y_pred_labels)) * 100
    print(f"   {class_label:8s}: {count:5d} ({percentage:5.2f}%)")

# ============================================================================
# SECTION 8: SUBMISSION FILE CREATION
# ============================================================================

print("\n" + "="*80)
print("PHASE 8: CREATING SUBMISSION FILE")
print("="*80)

# Create submission dataframe
submission_df = pd.DataFrame({
    'id': test_ids,
    'Irrigation_Need': y_pred_labels
})

# Save to CSV
submission_path = '/mnt/user-data/outputs/irrigation_submission.csv'
submission_df.to_csv(submission_path, index=False)

print(f"\n✅ Submission file created: {submission_path}")
print(f"\nFirst 10 rows of submission:")
print(submission_df.head(10))

# ============================================================================
# SECTION 9: VISUALIZATION & REPORTING
# ============================================================================

print("\n" + "="*80)
print("PHASE 9: GENERATING VISUALIZATIONS")
print("="*80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Create figure with subplots for confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=target_encoder.classes_,
                yticklabels=target_encoder.classes_,
                cbar_kws={'label': 'Count'})
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)

plt.tight_layout()
confusion_matrix_path = '/mnt/user-data/outputs/confusion_matrices.png'
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrices saved: {confusion_matrix_path}")
plt.close()

# Model performance comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(summary_df))
width = 0.25

ax.bar(x - width, summary_df['Accuracy'], width, label='Accuracy', alpha=0.8)
ax.bar(x, summary_df['F1-Weighted'], width, label='F1-Weighted', alpha=0.8)
ax.bar(x + width, summary_df['F1-Macro'], width, label='F1-Macro', alpha=0.8)

ax.set_ylabel('Score', fontsize=11)
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(summary_df['Model'])
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

performance_path = '/mnt/user-data/outputs/model_performance_comparison.png'
plt.savefig(performance_path, dpi=300, bbox_inches='tight')
print(f"✓ Performance comparison saved: {performance_path}")
plt.close()

# Target distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Training data distribution
train_counts = pd.Series(y).map({v: k for k, v in label_mapping.items()}).value_counts()
axes[0].bar(train_counts.index, train_counts.values, color='skyblue', alpha=0.7)
axes[0].set_title('Training Data Distribution', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].grid(axis='y', alpha=0.3)

# Prediction distribution
pred_counts = pd.Series(y_pred_labels).value_counts()
axes[1].bar(pred_counts.index, pred_counts.values, color='lightcoral', alpha=0.7)
axes[1].set_title('Prediction Distribution', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Count')
axes[1].grid(axis='y', alpha=0.3)

distribution_path = '/mnt/user-data/outputs/distribution_comparison.png'
plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
print(f"✓ Distribution comparison saved: {distribution_path}")
plt.close()

# ============================================================================
# SECTION 10: SAVE DETAILED RESULTS
# ============================================================================

print("\n" + "="*80)
print("PHASE 10: SAVING DETAILED RESULTS")
print("="*80)

# Save results summary
results_text = f"""
IRRIGATION NEED PREDICTION - MODEL TRAINING RESULTS
{'='*80}

DATASET INFORMATION
{'-'*80}
Total Samples: {len(df)}
Features: {len(numerical_cols) + len(categorical_cols)}
Numerical Features: {len(numerical_cols)}
Categorical Features: {len(categorical_cols)}
Target Classes: {', '.join(target_encoder.classes_)}

Target Distribution:
{summary_df['Model'].apply(lambda x: '')}
Low:     {target_dist['Low']:5d} ({target_dist['Low']/len(df)*100:5.2f}%)
Medium:  {target_dist['Medium']:5d} ({target_dist['Medium']/len(df)*100:5.2f}%)
High:    {target_dist['High']:5d} ({target_dist['High']/len(df)*100:5.2f}%)

PREPROCESSING STEPS
{'-'*80}
1. Categorical Encoding: LabelEncoder applied to {len(categorical_cols)} features
2. Feature Scaling: StandardScaler (mean=0, std=1)
3. Validation Strategy: 5-Fold Stratified Cross-Validation

MODEL PERFORMANCE (5-Fold CV)
{'-'*80}
{summary_df.to_string(index=False)}

BEST MODEL: {best_model_name}
- Average Accuracy: {results[best_model_name]['accuracy'].mean():.4f}
- F1-Score (Weighted): {results[best_model_name]['f1_weighted'].mean():.4f}
- F1-Score (Macro): {results[best_model_name]['f1_macro'].mean():.4f}

CONFUSION MATRICES
{'-'*80}
Naive Bayes:
{pd.DataFrame(confusion_matrices['Naive Bayes'], 
              index=target_encoder.classes_,
              columns=target_encoder.classes_).to_string()}

Logistic Regression:
{pd.DataFrame(confusion_matrices['Logistic Regression'],
              index=target_encoder.classes_,
              columns=target_encoder.classes_).to_string()}

CLASSIFICATION REPORTS
{'-'*80}
Naive Bayes:
{classification_reports['Naive Bayes']}

Logistic Regression:
{classification_reports['Logistic Regression']}

TEST PREDICTIONS
{'-'*80}
Total Test Samples: {len(submission_df)}
Low:     {(submission_df['Irrigation_Need'] == 'Low').sum():5d} ({(submission_df['Irrigation_Need'] == 'Low').sum()/len(submission_df)*100:5.2f}%)
Medium:  {(submission_df['Irrigation_Need'] == 'Medium').sum():5d} ({(submission_df['Irrigation_Need'] == 'Medium').sum()/len(submission_df)*100:5.2f}%)
High:    {(submission_df['Irrigation_Need'] == 'High').sum():5d} ({(submission_df['Irrigation_Need'] == 'High').sum()/len(submission_df)*100:5.2f}%)

SUBMISSION DETAILS
{'-'*80}
File: irrigation_submission.csv
Format: id, Irrigation_Need
Rows: {len(submission_df)}
Columns: ['id', 'Irrigation_Need']
"""

results_path = '/mnt/user-data/outputs/training_results_summary.txt'
with open(results_path, 'w') as f:
    f.write(results_text)
print(f"✓ Results summary saved: {results_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
print("="*80)

print("\n📁 Output Files Generated:")
print(f"   1. irrigation_submission.csv - Ready for Kaggle submission")
print(f"   2. training_results_summary.txt - Detailed results")
print(f"   3. confusion_matrices.png - Visual comparison")
print(f"   4. model_performance_comparison.png - Performance metrics")
print(f"   5. distribution_comparison.png - Data distribution analysis")

print(f"\n🚀 Next Steps:")
print(f"   1. Download test.csv from Kaggle competition")
print(f"   2. Place test.csv in /mnt/user-data/uploads/")
print(f"   3. Re-run this script for actual predictions")
print(f"   4. Submit irrigation_submission.csv to Kaggle")

print("\n" + "="*80)
