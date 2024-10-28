# pip install psycopg2
# pip install numpy

# Ignore SQLITE warnings related to Decimal numbers in the Chinook database
import warnings
warnings.filterwarnings('ignore')

# Import Dependencies
import numpy as np
import os
from flask import Flask, jsonify, render_template, request
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from sqlalchemy import func
from config import password
from config import db_host
import psycopg2
import joblib

# Read the database password from the environment variable in AWS for deployment
# password = os.getenv('DB_PASSWORD')
# db_host = os.getenv('DB_HOST')

#################################################
# Database Connection Setup
#################################################
# Create engine using the `postgres` database file on local host
engine = create_engine(f'postgresql://postgres:{password}@{db_host}:5432/Cancer_db')
# engine = create_engine(f'postgresql://postgres:{password}@host.docker.internal:5432/Cancer_db')
connection = engine.connect()

# Reflect Database into ORM classes
Base = automap_base()
Base.prepare(autoload_with=engine)
Base.classes.keys()

# Define the tables
cd = Base.classes.Cancer_Data

#################################################
# Flask Setup
#################################################

app = Flask(__name__)

#################################################
# Flask Routes
#################################################

@app.route("/")
def index():
    print("Server received request for 'Home/Index' page...")
    return render_template("index.html")

@app.route('/Concept_Ideation')
def ideation():
    print("Server received request for 'Project Concept and Ideation' page...")  
    return render_template("concept.html")

@app.route('/Data_API')
def integration():
    print("Server received request for 'Data fetching/API Integration' page...")  
    return render_template("api_data_integration.html")

@app.route('/Data_Pre-processing')
def visualisations():
    print("Server received request for 'Team Evaluatiion of Models' page...")  
    return render_template("data_pre-process.html")

@app.route("/Zahra_LR")
def logistic_regression():
    print("Server received request for 'Zahra's Logistic Regression Model' page...")
    return render_template("log_reg_model.html")

@app.route('/Jess_Aria_SVM')
def SVM():
    print("Server received request for 'Jess' and Aria's SVM Model' page...")  
    return render_template("SVM.html")

@app.route('/Jess_Aria_RF')
def RF():
    print("Server received request for 'Jess' and Aria's RF Model' page...")  
    return render_template("ran_for_model.html")

@app.route('/Cathy_and_Lauren_NN_KT')
def KT():
    print("Server received request for 'Cathy and Lauren's KT Model' page...")  
    return render_template("ker_tun_model.html")

@app.route('/Group_Eval')
def evaluation():
    print("Server received request for 'Team Evaluatiion of Models' page...")  
    return render_template("eval.html")

@app.route('/Model_Optimisation')
def optimisations():
    print("Server received request for 'Lauren Evaluatiion of Models' page...")  
    return render_template("optimise.html")

@app.route('/Prediction')
def prediction():
    # Step 3: Load the pre-trained logistic regression model and the refitted scaler
    return render_template('prediction_app.html')

@app.route('/Result', methods=['POST'])
def result():
    print("Server received request for 'Prediction Result' page...")

    try:   
    # Collect input data from the form for the selected 30 features
        input_features = [
            float(request.form['area_mean']),
            float(request.form['area_worst']),
            float(request.form['compactness_mean']),
            float(request.form['compactness_se']),
            float(request.form['compactness_worst']),
            float(request.form['concave points_mean']),
            float(request.form['concave points_se']),
            float(request.form['concave points_worst']),
            float(request.form['concavity_mean']),
            float(request.form['concavity_worst']),

            float(request.form['fractal_dimension_mean']),
            float(request.form['fractal_dimension_worst']),
            float(request.form['perimeter_mean']),
            float(request.form['perimeter_worst']),
            float(request.form['radius_mean']),
            float(request.form['radius_worst']),
            float(request.form['smoothness_mean']),
            float(request.form['smoothness_se']),
            float(request.form['smoothness_worst']),
            float(request.form['symmetry_mean']),

            float(request.form['symmetry_se']),
            float(request.form['symmetry_worst']),
            float(request.form['texture_mean']),
            float(request.form['texture_se']),
            float(request.form['texture_worst']),
            float(request.form['area_se']),
            float(request.form['concavity_se']),
            float(request.form['perimeter_se']),
            float(request.form['radius_se']),
            float(request.form['fractal_dimension_se']) 
        ]

        # Convert input features into a NumPy array with shape (1, 32)
        features = np.array(input_features).reshape(1, -1)
        # Step 4: Scale the input features using the new refitted scaler
        
        # scaled_features = scaler.transform(features)
        # Model 1: Load the pre-trained Logistic Regression Model
        model_1 = joblib.load('Saved_Models/tuned_logistic_regression_model.pkl')
        # scaler = joblib.load('Saved_Models/scaler.pkl')
        # Model 2: Load the pre-trained SVM
        model_2 = joblib.load('Saved_Models/tunned_model_svm.pkl')
        # Model 3: Load the pre-trained Random Forest
        model_3 = joblib.load('Saved_Models/rf_model.pkl')
        # Model 4: Load the pre-trained Keras Tuner
        model_4 = joblib.load('Saved_Models/kt_best_model.pkl')
        # Model 5: Load the pre-trained Ensemble Method



        # Make a prediction using the scaled features
        prediction_1 = model_1.predict(features)
        prediction_2 = model_2.predict(features)
        prediction_3 = model_3.predict(features)
        prediction_4 = model_4.predict(features)
        # prediction_5 = model_5.predict(model_5)
        
        # Optionally, get the predicted class or probability
        # Get the predicted class or probability
        # Prepare predictions text

        log_pred_text_1 = "Malignant" if prediction_1[0] == 1 else "Benign"  # Assuming binary classification
        log_pred_text_2 = "Malignant" if prediction_2[0] == 1 else "Benign"  # Assuming binary classification
        log_pred_text_3 = "Malignant" if prediction_3[0] == 1 else "Benign"  # Assuming binary classification
        log_pred_text_4 = "Malignant" if prediction_4[0] == 1 else "Benign"  # Assuming binary classification
        # log_pred_text_1 = "Malignant" if log_prediction[0] == 1 else "Benign"  # Assuming binary classification

        # predictions_text = {
        #     'Logistic Regression': f"Class Value: {prediction_1[0]} - Label: {'Malignant' if prediction_1[0] == 1 else 'Benign'}",
        #     'SVM': f"Class Value: {prediction_2[0]} - Label: {'Malignant' if prediction_2[0] == 1 else 'Benign'}",
        #     'Random Forest': f"Class Value: {prediction_3[0]} - Label: {'Malignant' if prediction_3[0] == 1 else 'Benign'}",
        #     'Keras Tuner': f"Class Value: {prediction_4[0]} - Label: {'Malignant' if prediction_4[0] == 1 else 'Benign'}",
        # }

        # log_pred_text_5 = "Malignant" if prediction_5[0] == 1 else "Benign"  # Assuming binary classification
        # print(predictions_text)
        # Render the result page with the prediction
        return render_template('result.html', log_pred=(
            f'Logistic Regression Class Prediction: {prediction_1[0]} = {log_pred_text_1}'
            f'SVM Class Prediction: {prediction_2[0]} = {log_pred_text_2}'
            f'SVM Class Prediction: {prediction_3[0]} = {log_pred_text_3}'
            f'SVM Class Prediction: {prediction_4[0]} = {log_pred_text_4}'
        ))

        #      return render_template('result.html', predictions=f'{
        #     'Logistic Regression': f"Class Value: {prediction_1[0]} - Label: {'Malignant' if prediction_1[0] == 1 else 'Benign'}",
        #     'SVM': f"Class Value: {prediction_2[0]} - Label: {'Malignant' if prediction_2[0] == 1 else 'Benign'}",
        #     'Random Forest': f"Class Value: {prediction_3[0]} - Label: {'Malignant' if prediction_3[0] == 1 else 'Benign'}",
        #     'Keras Tuner': f"Class Value: {prediction_4[0]} - Label: {'Malignant' if prediction_4[0] == 1 else 'Benign'}",
        # }')
        

    except ValueError as e:
        return render_template('result.html', prediction_text=f'The predicted class is not determined.')
    except Exception as e:
        return render_template('result.html', prediction_text=f'The predicted class is not determined.')

@app.route('/Conclude')
def conclusion():
    print("Server received request for 'Conclusion' page...")  
    return render_template("conclude.html")

@app.route("/api_json_connectivity")
def get_cancer_db_data():
    print("Server received request for fetched 'API Jsonified database data'...")
    session = Session(engine)

    try:
        query = text("""
        SELECT json_agg(row_to_json("Cancer_Data")) FROM "Cancer_Data";
        """)
        # Execute the query
        results = session.execute(query)
        # all_data = results.scalar()  # Get the first column of the first row
    
        # # Convert list of tuples into normal list
        # all_data = list(np.ravel(results))
        all_data = results.fetchone()[0]  # Get the first row and first column
        return jsonify(all_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

if __name__ == "__main__":
    app.run(debug=True) #local deployment
    # app.run(host='0.0.0.0', port=8080, debug=True) #internet deployment