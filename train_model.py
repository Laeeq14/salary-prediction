import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
data = pd.read_csv('Salary_Data.csv')

# Remove rows with missing values
data_clean = data.dropna()

# Define the features and target
features = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
target = 'Salary'

X = data_clean[features]
y = data_clean[target]

# Define the column transformer with one-hot encoding for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'Education Level', 'Job Title']),
        ('num', 'passthrough', ['Age', 'Years of Experience'])
    ]
)

# Create a pipeline with the preprocessor and a linear regression model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Save the trained model
joblib_file = 'salary_model.pkl'
joblib.dump(model_pipeline, joblib_file)

# Check the model performance on the test set
test_score = model_pipeline.score(X_test, y_test)
print(f"Test Score: {test_score}")
