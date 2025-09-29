import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Dataset
data = [
    ["Swiggy 250", "Food"],
    ["Uber 300", "Travel"],
    ["Amazon 1200", "Shopping"],
    ["Groceries 450", "Groceries"],
    ["Netflix 500", "Entertainment"],
    ["Stationery 100", "Education"],
    ["Book 250", "Education"],
    ["Gym 700", "Health"],
    ["Bus 50", "Travel"],
    ["Coffee 150", "Food"],
]

df = pd.DataFrame(data, columns=["Transaction", "Category"])
print("Dataset preview:")
print(df.head())

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
samples = ["Swiggy 300", "Bus 70", "Book 300"]
sample_tfidf = vectorizer.transform(samples)
predictions = model.predict(sample_tfidf)
for t, p in zip(samples, predictions):
    print(f"{t} → {p}")
