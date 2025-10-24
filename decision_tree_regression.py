#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

#Create Dataset
data = {
    'X' : [1,2,3,4,5],
    'y' : [2,3,2,10,12]

}
df = pd.DataFrame(data)

#Display dataset
print("Dataset:")
print(df)

#Calculate parent variance
parent_variance = np.var(df['y'])
print(f"\nParent Variance: {parent_variance:.3f}")

#Manual split to see variance reduction
split_point = 3
left_y = df[df['X'] <= split_point]['y']
right_y = df[df['X'] > split_point]['y']

left_var = np.var(left_y)
right_var = np.var(right_y)

n_left = len(left_y)
n_right = len(right_y)
n_total = len(df)

weighted_child_var = (n_left / n_total) * left_var + (n_right / n_total) * right_var
variance_reduction = parent_variance - weighted_child_var

print(f"\nAfter Split (X <= {split_point}):")
print(f"Left Variance: {left_var:.3f}, Right Variance: {right_var:.3f}")
print(f"Weighted Child Variance: {weighted_child_var:.3f}")
print(f"Variance Reduction: {variance_reduction:.3f}")

#Train Decision Tree Regressor
X = df[['X']]
y = df['y']

model = DecisionTreeRegressor(random_state= 42)
model.fit(X,y)

#Predict & Visualize tree
pred = model.predict(X)

plt.figure(figsize=(8,5))
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, pred, color='red', label='Predicted', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Decision Tree Regression - Variance Reduction Intuition')
plt.legend()
plt.show()

#Plot tree structure
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=['X'], filled=True, rounded=True)
plt.title("Decision Tree Structure (Regression)")
plt.show()

#Print Predicitons
print("\nPredictions:")
for xi, yi, pi in zip(df['X'], df['y'], pred):
    print(f"X={xi} | Actual y={yi} | Predicted y={pi:.2f}")
