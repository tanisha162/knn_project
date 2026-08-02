import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data: movie budget/runtime vs good/bad
data = {
    "budget": [50, 55, 10, 8, 60, 12, 48, 15, 52, 9],
    "runtime": [120, 115, 90, 85, 125, 95, 118, 88, 122, 92],
    "good_movie": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
print(df)

X = df[["budget", "runtime"]]
y = df["good_movie"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features - critical for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = KNeighborsClassifier(n_neighbors=4)
model.fit(X_train_scaled, y_train)

# Predict
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

print("\nPredictions:", predictions)
print("Actual:", y_test.values)
print("Accuracy:", accuracy)

# Save results
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})
results_df.to_csv("results.csv", index=False)
print("\nSaved results.csv")