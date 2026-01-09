# Spam Email Detection - Machine Learning Project

## Project Overview
This project implements a **Logistic Regression-based spam email classifier** that analyzes emails and predicts whether they are legitimate or spam based on extracted features.

---

## Dataset
- **File**: `a_bliadze25_42198.csv`
- **Delimiter**: Comma-separated values (CSV)
- **Features Used**:
  - `words`: Number of words in the email
  - `links`: Number of URLs/links in the email
  - `capital_words`: Count of consecutive capital letters (e.g., "SPAM", "WIN")
  - `spam_word_count`: Frequency of spam keywords ('free', 'click', 'winner', 'prize', 'urgent', 'buy', 'limited', 'act now')
- **Target Variable**: `is_spam` (0 = Legitimate, 1 = Spam)

---

## Step-by-Step Workflow

### **Step 1: Import Required Libraries**
The project imports essential Python libraries for data analysis, visualization, and machine learning:

```python
import pandas as pd                    # Data manipulation and analysis
import matplotlib.pyplot as plt        # Data visualization
import seaborn as sns                  # Advanced statistical visualization
import numpy as np                     # Numerical computing
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.linear_model import LogisticRegression   # ML algorithm
from sklearn.metrics import confusion_matrix, accuracy_score  # Model evaluation
import re                              # Regular expressions for text processing
```

---

### **Step 2: Load and Explore Data**
Load the CSV dataset and examine the first few rows to understand the data structure:

```python
df = pd.read_csv('a_bliadze25_42198.csv', delimiter=',')
df.head()  # Display first 5 rows
```

**Output**: Shows the structure of features and target variable.

---

### **Step 3: Feature Selection and Target Definition**
Select input features (X) and target variable (y):

```python
X = df[['words', 'links', 'capital_words', 'spam_word_count']]  # Features
y = df['is_spam']  # Target (0 or 1)
```

---

### **Step 4: Data Splitting**
Split the dataset into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

- **Training Set**: 70% of data (used to train the model)
- **Testing Set**: 30% of data (used to evaluate model performance on unseen data)
- **Random State**: 42 (ensures reproducibility)

---

### **Step 5: Train Logistic Regression Model**
Create and train the Logistic Regression classifier:

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

**Output**: The model learns the relationship between features and spam classification.

---

### **Step 6: Display Model Coefficients**
Print the learned coefficients for each feature:

```python
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef}")
print(f"Intercept: {model.intercept_[0]}")
```

**Interpretation**:
- **Positive coefficient**: Increases likelihood of being classified as spam
- **Negative coefficient**: Decreases likelihood of being classified as spam
- **Intercept**: The bias term in the logistic regression equation

---

### **Step 7: Calculate Training and Testing Accuracy**
Evaluate model performance on both training and testing data:

```python
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")
```

---

### **Step 8: Generate Predictions**
Make predictions on the test set:

```python
y_pred = model.predict(X_test)
```

**Note**: Predictions are binary (0 = Not Spam, 1 = Spam)

---

### **Step 9: Confusion Matrix**
Evaluate model classification performance:

```python
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
```

**Matrix Structure**:
```
                 Predicted Legitimate  Predicted Spam
Actual Legitimate        TN                 FP
Actual Spam              FN                 TP
```

Where:
- **TN** (True Negatives): Correctly identified legitimate emails
- **TP** (True Positives): Correctly identified spam emails
- **FP** (False Positives): Legitimate emails incorrectly classified as spam
- **FN** (False Negatives): Spam emails incorrectly classified as legitimate

---

### **Step 10: Calculate Accuracy Score**
Compute the overall accuracy of the model:

```python
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on Test Set: {accuracy:.4f}")
```

**Formula**: Accuracy = (TP + TN) / (TP + TN + FP + FN)

---

### **Step 11: Feature Extraction Function**
Define a function to extract features from any email text:

```python
def extract_features(email_text):
    words = len(email_text.split())
    links = len(re.findall(r'http[s]?://\S+', email_text))
    capital_words = len(re.findall(r'\b[A-Z]{2,}\b', email_text))
    spam_keywords = ['free', 'click', 'winner', 'prize', 'urgent', 'buy', 'limited', 'act now']
    spam_word_count = sum(email_text.lower().count(keyword) for keyword in spam_keywords)
    
    return pd.DataFrame([[words, links, capital_words, spam_word_count]], 
                       columns=['words', 'links', 'capital_words', 'spam_word_count'])
```

**Features Extracted**:
1. **Words**: Total word count
2. **Links**: Count of HTTP/HTTPS URLs
3. **Capital Words**: Count of consecutive capital letter sequences (2+ letters)
4. **Spam Word Count**: Frequency of known spam keywords

---

### **Step 12: Email Classification Function**
Create a function to classify emails and display predictions:

```python
def check_email(email_text):
    features = extract_features(email_text)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    print("\n" + "="*50)
    print("EMAIL SPAM PREDICTION")
    print("="*50)
    print(f"Extracted Features:")
    print(f"  Words: {features['words'].values[0]}")
    print(f"  Links: {features['links'].values[0]}")
    print(f"  Capital Words: {features['capital_words'].values[0]}")
    print(f"  Spam Word Count: {features['spam_word_count'].values[0]}")
    print(f"\nPrediction: {'SPAM' if prediction == 1 else 'NOT SPAM'}")
    print(f"Confidence: {max(probability)*100:.2f}%")
    print("="*50)
```

---

### **Step 13: Test with Sample Emails**

#### **Legitimate Email Example**
```python
email_legitimate = "Hi Sarah, I hope you are doing well. I wanted to follow up on our meeting last week about the project proposal. Could you please send me the updated timeline and budget details when you have a chance? Thank you for your help. Best regards, John"

check_email(email_legitimate)
```

**Expected Output**: NOT SPAM (low count of spam keywords, normal capitalization)

#### **Spam Email Example**
```python
email_spam = "CLICK HERE NOW! FREE PRIZE WINNER! You have been selected to WIN BIG! Limited time offer buy now act now! FREE money waiting for you! CLICK CLICK CLICK!"

check_email(email_spam)
```

**Expected Output**: SPAM (high count of spam keywords, excessive capitalization)

---

### **Step 14: Visualization 1 - Class Distribution (Pie Chart)**
Visualize the balance of spam vs. legitimate emails:

```python
plt.figure(figsize=(8, 6))
class_counts = y.value_counts()
colors = ['#2ecc71', '#e74c3c']
plt.pie(class_counts, labels=['Legitimate', 'Spam'], autopct='%1.1f%%', 
        colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
plt.title('Class Distribution: Spam vs Legitimate Emails', fontsize=14, fontweight='bold', pad=20)
plt.axis('equal')
plt.show()
```

**Explanation**: This pie chart shows the proportion of spam and legitimate emails in the dataset. A balanced or imbalanced distribution affects model training and performance.

---

### **Step 15: Visualization 2 - Confusion Matrix Heatmap**
Display model performance using a heatmap:

```python
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Legitimate', 'Spam'], 
            yticklabels=['Legitimate', 'Spam'],
            annot_kws={'fontsize': 14, 'weight': 'bold'})
plt.title('Confusion Matrix - Model Performance on Test Set', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Actual Class', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
plt.show()
```

**Explanation**: This heatmap visualizes the model's classification performance on unseen test data, showing true positives, true negatives, false positives, and false negatives for each class.

---

### **Step 16: Visualization 3 - Feature Importance (Bar Chart)**
Visualize how each feature influences spam classification:

```python
plt.figure(figsize=(10, 6))
features = X.columns
coefficients = model.coef_[0]
colors_bar = ['#3498db' if c > 0 else '#e74c3c' for c in coefficients]
bars = plt.bar(features, coefficients, color=colors_bar, edgecolor='black', linewidth=1.5)
plt.title('Feature Importance - Logistic Regression Coefficients', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Coefficient Value', fontsize=12, fontweight='bold')
plt.xlabel('Features', fontsize=12, fontweight='bold')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(axis='y', alpha=0.3)
plt.legend(['Positive Impact on Spam', 'Negative Impact on Spam'], loc='upper left')
plt.show()
```

**Explanation**: This bar chart displays the coefficients learned by the logistic regression model. Positive coefficients increase spam likelihood, while negative coefficients decrease it. Features like "words" and "capital_words" are typically strong spam indicators.

---

## Model Performance Metrics

| Metric | Value |
|--------|-------|
| **Training Accuracy** | See output from Step 7 |
| **Testing Accuracy** | See output from Step 7 |
| **True Positives (TP)** | Correctly identified spam emails |
| **True Negatives (TN)** | Correctly identified legitimate emails |
| **False Positives (FP)** | Legitimate emails flagged as spam |
| **False Negatives (FN)** | Spam emails classified as legitimate |

---

## Key Takeaways

1. **Data Preparation**: Features are extracted from email text to represent numerical characteristics.
2. **Model Training**: Logistic Regression learns the relationship between features and spam classification.
3. **Evaluation**: Confusion matrix and accuracy metrics measure model performance.
4. **Prediction**: The trained model can classify new emails as spam or legitimate.
5. **Visualization**: Charts provide insights into data distribution and feature importance.
6. **Feature Engineering**: Spam keyword detection and capitalization counts are effective indicators of spam emails.

---

## Files Required
- `a_bliadze25_42198.csv` - Dataset with email features and spam labels
- `spam_email_detection.ipynb` - Jupyter Notebook with complete analysis
- `correlation.ipynb` - Correlation analysis notebook

---

## Libraries Used
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and metrics
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical visualization
- **NumPy**: Numerical computing
- **Regular Expressions (re)**: Text pattern matching

---

## How to Run
1. Open `spam_email_detection.ipynb` in Jupyter Notebook or VS Code
2. Ensure `a_bliadze25_42198.csv` is in the same directory
3. Execute cells sequentially from top to bottom
4. View outputs, visualizations, and model predictions

---

**Project Created**: AI/ML Mid-Year Assignment 2026  
**Student ID**: a_bliadze25  
**Repository**: https://github.com/Ani-Bliadze/aimlmid2026_a_bliadze25
