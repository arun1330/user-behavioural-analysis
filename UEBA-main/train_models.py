# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import IsolationForest
# import tensorflow as tf
# from tensorflow.keras import layers, models

# # Load the dataset
# data = pd.read_csv('data/employee_data.csv')

# # Define features and target
# features = ['employee_id', 'hours_worked', 'tasks_completed', 'department']
# numeric_features = ['hours_worked', 'tasks_completed']
# categorical_features = ['employee_id', 'department']

# X = data[features]
# y = data['anomaly']

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define preprocessor
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numeric_features),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#     ]
# )

# # Fit the preprocessor on the training data
# preprocessor.fit(X_train)

# # Save the preprocessor
# joblib.dump(preprocessor, 'models/preprocessor.pkl')

# # Preprocess the training data
# X_train_processed = preprocessor.transform(X_train)

# # Train Isolation Forest
# iso_forest = IsolationForest(contamination=0.05, random_state=42)
# iso_forest.fit(X_train_processed)

# # Save the model
# joblib.dump(iso_forest, 'models/isolation_forest_model.pkl')

# # Define Autoencoder
# input_dim = X_train_processed.shape[1]
# encoding_dim = 14  # Change as necessary

# input_layer = layers.Input(shape=(input_dim,))
# encoder = layers.Dense(encoding_dim, activation="relu")(input_layer)
# decoder = layers.Dense(input_dim, activation="sigmoid")(encoder)
# autoencoder = models.Model(inputs=input_layer, outputs=decoder)

# # Compile the model
# autoencoder.compile(optimizer='adam', loss='mse')

# # Train the model
# autoencoder.fit(X_train_processed, X_train_processed, epochs=50, batch_size=32, validation_split=0.2)

# # Save the model
# autoencoder.save('models/autoencoder_model.h5')



import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the datasets
employee_data = pd.read_csv('data/employee_data.csv')
currency_data = pd.read_csv('data/currency.csv')

# Employee Data Analysis
# Define features and target for employee data
employee_features = ['employee_id', 'hours_worked', 'tasks_completed', 'department']
numeric_features = ['hours_worked', 'tasks_completed']
categorical_features = ['employee_id', 'department']

X_employee = employee_data[employee_features]
y_employee = employee_data['anomaly']

# Split the data
X_train_emp, X_test_emp, y_train_emp, y_test_emp = train_test_split(X_employee, y_employee, test_size=0.2, random_state=42)

# Define preprocessor
preprocessor_emp = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Fit the preprocessor on the training data
preprocessor_emp.fit(X_train_emp)

# Save the preprocessor
joblib.dump(preprocessor_emp, 'models/preprocessor_emp.pkl')

# Preprocess the training data
X_train_emp_processed = preprocessor_emp.transform(X_train_emp)

# Train Isolation Forest
iso_forest_emp = IsolationForest(contamination=0.05, random_state=42)
iso_forest_emp.fit(X_train_emp_processed)

# Save the model
joblib.dump(iso_forest_emp, 'models/isolation_forest_emp_model.pkl')

# Define Autoencoder for employee data
input_dim_emp = X_train_emp_processed.shape[1]
encoding_dim_emp = 14

input_layer_emp = layers.Input(shape=(input_dim_emp,))
encoder_emp = layers.Dense(encoding_dim_emp, activation="relu")(input_layer_emp)
decoder_emp = layers.Dense(input_dim_emp, activation="sigmoid")(encoder_emp)
autoencoder_emp = models.Model(inputs=input_layer_emp, outputs=decoder_emp)

# Compile the model
autoencoder_emp.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder_emp.fit(X_train_emp_processed, X_train_emp_processed, epochs=50, batch_size=32, validation_split=0.2)

# Save the model
autoencoder_emp.save('models/autoencoder_emp_model.h5')

# Currency Data Analysis
# Example processing of currency data
# You can customize this section depending on what analysis you want to perform
currency_summary = currency_data.describe()

# Print summary of currency data
print("Currency Data Summary:")
print(currency_summary)

# If you need to process and save the currency data model, you can add additional code here

# Example: Save the currency data summary as a CSV
currency_summary.to_csv('data/currency_data_summary.csv')

