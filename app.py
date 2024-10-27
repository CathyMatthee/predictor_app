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
    return render_template("lin_reg_model.html")

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

        # Convert input features into a NumPy array with shape (1, 30)
        features = np.array(input_features).reshape(1, -1)
        # Step 3: Load the pre-trained logistic regression model
        model = joblib.load('Logistic_Regression_Model/FeatureReduced_logistic_regression_model.pkl')
        scaler = joblib.load('Logistic_Regression_Model/scaler_reduced.pkl')  # Load the new scaler fitted on 11 features

        # Step 4: Scale the input features using the new refitted scaler
        scaled_features = scaler.transform(features)

        # Make a prediction using the scaled features
        prediction = model.predict(scaled_features)

        # Optionally, get the predicted class or probability
        prediction_text = "Malignant" if prediction[0] == 1 else "Benign"  # Assuming binary classification

        # Render the result page with the prediction
        return render_template('result.html', prediction_text=f'The logistic regression model predicted class is: {prediction[0]} which is {prediction_text}')
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