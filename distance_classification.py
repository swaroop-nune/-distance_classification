import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import wandb

# Initialize WandB
wandb.init(project="distance_classification_project")

# Generate Dummy Data
np.random.seed(42)
X = np.random.rand(100, 2)  # 100 samples, 2 features
y = np.random.choice([0, 1], size=100)  # Binary labels

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate Model
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Log Metrics
wandb.log({"accuracy": accuracy})
