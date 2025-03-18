from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd

# Load dataset
data = pd.read_csv('data.csv')

# Preprocess data
# ...

# Split data into features (X) and target (y)
X = data.drop('target', axis=1)
y = data['target']

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Choose model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))