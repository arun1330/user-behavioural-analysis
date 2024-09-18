import tensorflow as tf
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import joblib
import os
import logging
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
db = SQLAlchemy(app)

logging.basicConfig(level=logging.DEBUG)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(150))
    last_name = db.Column(db.String(150))
    email = db.Column(db.String(150), unique=True, nullable=False)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

iso_forest = joblib.load('models/isolation_forest_model.pkl')
autoencoder = tf.keras.models.load_model('models/autoencoder_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
preprocessor = joblib.load('models/preprocessor.pkl')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials, please try again.')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')

        if not (first_name and last_name and email and username and password):
            return 'Please fill out all fields.'

        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            return 'Email already registered. Please use a different email.'

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return 'Username already exists. Please choose a different username.'

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(first_name=first_name, last_name=last_name, email=email, username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        username = session['username']
        return render_template('dashboard.html', username=username)
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/results', methods=['GET', 'POST'])
def results():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            data = pd.read_csv(file_path)
            processed_data = preprocessor.transform(data)

            iso_predictions = iso_forest.predict(processed_data)
            anomalies = iso_predictions == -1

            reconstructions = autoencoder.predict(processed_data)
            reconstruction_errors = np.mean(np.square(processed_data - reconstructions), axis=1)
            auto_threshold = 0.1
            auto_anomalies = reconstruction_errors > auto_threshold

            threat_threshold = 0.2
            threats = reconstruction_errors > threat_threshold

            combined_anomalies = anomalies | auto_anomalies
            combined_threats = combined_anomalies & threats

            total_anomalies = np.sum(combined_anomalies)
            total_threats = np.sum(combined_threats)

            return render_template('results.html', anomalies=total_anomalies, threats=total_threats)

    return render_template('results.html', anomalies=None, threats=None)




@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Store the file path in session
            session['file_path'] = file_path
            
            return redirect(url_for('result'))
    
    return render_template('upload.html')


@app.route('/result', methods=['GET'])
def result():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    file_path = session.get('file_path')
    if file_path is None:
        flash('No file uploaded')
        return redirect(url_for('upload'))

    # Process the CSV file and generate results
    dataset_info, first_5_rows, missing_values, confusion_mat, class_report, accuracy, plot_path = process_credit_card_data(file_path)

    return render_template('result.html',
                           dataset_info=dataset_info,
                           first_5_rows=first_5_rows,
                           missing_values=missing_values,
                           confusion_mat=confusion_mat,
                           class_report=class_report,
                           accuracy=accuracy,
                           plot_path=plot_path)


def process_credit_card_data(file_path):
    df = pd.read_csv(file_path)

    # Explore the data
    buffer = io.StringIO()
    df.info(buf=buffer)
    dataset_info = buffer.getvalue()

    first_5_rows = df.head().to_html()
    missing_values_df = df.isnull().sum().to_frame(name='Missing Values')
    missing_values = missing_values_df.to_dict()['Missing Values']

    # Preprocess the data
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    # Drop the original 'Amount' and 'Time' columns
    df = df.drop(columns=['Amount', 'Time'])

    # Define the feature variables (X) and the target variable (y)
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    confusion_mat = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    # Save Feature Importances Plot
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()

    plot_path = os.path.join(static_dir, 'feature_importances.png')
    plt.savefig(plot_path)
    plt.close()

    return dataset_info, first_5_rows, missing_values, confusion_mat, class_report, accuracy, 'feature_importances.png'



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)