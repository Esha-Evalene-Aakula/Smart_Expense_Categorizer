import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
print(df)


# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["Transaction"], df["Category"], test_size=0.3, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Test samples
samples = ["Swiggy 300", "Bus 55", "Book 300"]
sample_tfidf = vectorizer.transform(samples)
predictions = model.predict(sample_tfidf)
for t, p in zip(samples, predictions):
    print(f"{t} → {p}")
