# **User Behavioural Analysis in an Autonomous System**

### **Aims:**

  - The primary aim of this script is to build an anomaly detection and fraud detection system using machine learning models that can be integrated into a web application (Flask) to automatically detect unusual behaviors or fraudulent activities from employee data. 

  - The application allows users to upload datasets and analyze them for anomalies using both Isolation Forest and Autoencoder models. By doing so, it helps in identifying potential security threats or fraud within the organization's operations.

### **Objectives:**

##### **1. Data Collection and Preprocessing:**
   
  - Load and process employee data to make it suitable for machine learning models.
    
  - Normalize numeric data and encode categorical data to create a consistent and robust input format.

##### **2. Anomaly Detection:**

  - Build and train an Isolation Forest model to detect anomalous data points based on behavioral patterns.
  
  - Implement an Autoencoder neural network to capture normal patterns and detect reconstruction errors, which could indicate anomalies.

##### **3. Integration into Flask Web Application:**

  - Develop a user interface using Flask where users can upload data and view the anomaly and fraud detection results.
    
  - Provide authentication and secure access for users, ensuring that only authorized individuals can access the application.
  
  - Save the trained models and preprocessing pipelines to make the system reusable and scalable for new datasets.

##### **4. Anomaly and Fraudulent Activity Detection:**
 
  - Combine the results of the Isolation Forest and Autoencoder models to provide an aggregated view of anomalies, with an additional focus on potential fraudulent activities by identifying anomalies that cross certain thresholds.

##### **5. Ongoing Maintenance:**

  - Conduct regular reviews of alert configurations and thresholds.
  
  - Update the system based on network traffic patterns and new threat intelligence.
  
  - Establish a feedback loop with administrators to continually improve the system.

### **Tools and Techniques Used:**

##### **1. Flask Web Framework:**

- **Flask:** A lightweight web framework is used to build the user interface for this anomaly detection system. Flask allows easy routing, templating (via render_template), and handling of HTTP requests (such as POST for file uploads).

- **Flask SQLAlchemy:** This is used for database management, allowing secure storage of user credentials and session management for logged-in users.

- **Flask-Werkzeug:** This package handles secure password hashing (generate_password_hash) and secure file uploads (secure_filename).

##### **2. Data Preprocessing:**

- **Pandas:** The script uses Pandas to load and manage the dataset, especially for data cleaning and manipulation (e.g., selecting relevant features).

- **Scikit-learn’s ColumnTransformer:** This is used to preprocess the data by applying transformations to both numeric and categorical data.

- **StandardScaler:** Normalizes numeric features such as hours_worked and tasks_completed so that they follow a standard distribution.

- **OneHotEncoder:** Encodes categorical features (employee_id, department) into binary formats to allow machine learning algorithms to process them efficiently.

- **Train-Test Split:** The train_test_split function from Scikit-learn divides the data into training and testing sets to evaluate model performance.

##### **3. Anomaly Detection Models:**

- **Isolation Forest:** The Isolation Forest algorithm is an unsupervised learning technique that isolates anomalies by randomly selecting features and splitting the data. It is particularly useful for detecting outliers that represent fraudulent or suspicious activities.

- **Contamination Parameter:** This parameter is set to 0.05, meaning the model assumes 5% of the data could be anomalous.

- The model is saved using Joblib to allow future use without retraining.

- **Autoencoder (Neural Network):** TensorFlow and Keras: These libraries are used to build an Autoencoder model, which learns to compress the input data into a latent space (encoding) and then reconstruct it back (decoding).

- The autoencoder aims to minimize the reconstruction error, and unusually high errors could indicate anomalies in the input data.

- **Model Architecture:** The input layer accepts the preprocessed features.

- The encoding layer compresses the data into a lower-dimensional representation (14 dimensions in this case).

- The decoding layer reconstructs the original input.

- **Loss Function:** Mean Squared Error (MSE) is used as the loss function to minimize the reconstruction difference.

- The model is trained for 50 epochs, and the result is saved in H5 format (autoencoder_model.h5).

##### **4. File Handling and Security:**

- **Secure File Uploads:** The function allowed_file checks if the uploaded file is a CSV format, ensuring secure handling of user inputs.

- **Flask Sessions:** Secure sessions are implemented to handle user authentication, ensuring only authorized users can access the system. Passwords are hashed using PBKDF2-SHA256 for secure storage.
  
##### **5. Deployment and Model Management:**

- **Joblib:** Used for saving and loading both the preprocessor and the Isolation Forest model, ensuring that the application can preprocess new datasets and run predictions without re-training the model.

- **TensorFlow Keras:** The autoencoder is saved as an H5 file to be loaded later for prediction.

