
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

tree_ = model.tree_
print(f"Values shape: {tree_.value.shape}")
print(f"Root node value (raw): {tree_.value[0]}")
print(f"Root node value (int cast): {tree_.value[0].astype(int)}")
print(f"Root node value (list): {tree_.value[0].astype(int).tolist()}")

# Check if any value is normalized
print(f"Max value in tree: {np.max(tree_.value)}")
print(f"Samples at root: {tree_.n_node_samples[0]}")
reconstructed = tree_.value[0] * tree_.n_node_samples[0]
print(f"Reconstructed counts: {reconstructed}")
