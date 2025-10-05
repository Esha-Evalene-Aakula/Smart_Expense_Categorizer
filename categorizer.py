# expense_categorizer.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Dataset
data = {
    'text': [
        'Swiggy 300', 'Zomato 250', 'Dominos 500',
        'Uber 200', 'Ola 150', 'Bus 70', 'Train 120',
        'Book 300', 'Tuition 2000', 'Course 500',
        'Electricity Bill 700', 'Rent 5000', 'Groceries 1200', 'Movie 350'
    ],
    'category': [
        'Food', 'Food', 'Food',
        'Travel', 'Travel', 'Travel', 'Travel',
        'Education', 'Education', 'Education',
        'Bills', 'Bills', 'Groceries', 'Entertainment'
    ]
}

df = pd.DataFrame(data)
df['text'] = df['text'].str.lower()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42)

# TF-IDF + Logistic Regression
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict custom transactions
test_samples = ['Swiggy 300', 'Bus 70', 'Book 300']
test_samples = [t.lower() for t in test_samples]
X_test_custom = vectorizer.transform(test_samples)
predictions = model.predict(X_test_custom)

print("\nSample Predictions:\n")
for text, label in zip(test_samples, predictions):
    print(f"{text} → {label}")
