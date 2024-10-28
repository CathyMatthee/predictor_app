### Project Proposal

# Goal of the Project:
The purpose for this project is to assist with breast cancer survial through early intervention. A website will be created for the use of doctors which will contain user inputted data based on the visual characteristics of a cancer. A machine learning model will be used to predict a diagnosis on whether the identified cancer is Benign or Malignant based on these visual characteristics. The features that will be used to assist in determining the diagnosis include: mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean compactness, mean concavity and mean concave points. The dataset identifed for this project is the Breast Cancer Wisconsin (Diagnostic) Data Set. 

# Dataset Link: 
https://www.kaggle.com/datasets/erdemtaha/cancer-data/data

# Slide Deck:
https://www.canva.com/design/DAGUqhcvQm0/zgpipBtmAcZnmDA6aaSEtw/edit?utm_content=DAGUqhcvQm0&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

# Inspiration:
<img src="https://github.com/CathyMatthee/predictor_app/blob/main/images/proposal.png" alt="Proposal Inspiration" width="300"/>

# Method
The data will undergo cleaning and removal of any duplications or unnecessary data. Feature engineering will then be conducted on the dataset to ensure the data is in a usable state for machine learning. Postgres will be used to manage the database and the app.py flask app will be used to connect to the database to enable a website to call information from it. A number of machine learning models will be tested and evaluated on their prediction of the cancer diagnosis. The models include logistic regression, SVM, random forest and potentially Deep Neural Network. Each model will undergo optimisation and then they will be evaluated to select the best model. The selected machine learning model will then be used in the final website. The website will look similar to the above screenshot to enable a user to alter different visual characteristics and the model will output a predicted diagnosis.

# Licensing:
This Data has a CC BY-NC-SA 4.0 License.
https://creativecommons.org/licenses/by-nc-sa/4.0/

# Ethics:
This dataset contains a unique anonymous ID number for each patients cancer data. This is not considered a personally identifiable information as it is not linked back to any specific personal information of that patient such as a drivers license number or social security number.

# Workflow for Reproducibility

## Data Fetching and API Integration

This section explains how to connect to a PostgreSQL database using psycopg2 in python.

### Database Setup (pgAdmin4)
1. Download/clone the all the files from dataset from https://github.com/lhenry97/Group_1-Project_4.git.
2. Open pgAdmin 4 and create a new database called Cancer_db
3. Right click on Cancer_db and select Query Tool
4. Select the Folder icon to open a file called "cancer_data.sql" from the data folder
5. Run the "create table" query to load the table columns
6. Refresh the Table in Schemas, right-click on Cancer_Data table and select "Import/Export data"
7. Find the file called Cancer_Data.csv also in the "data" folder and open this file.
8. In Options menu set Header to "active" and Delimiter as ",".
9. Optionally, run the json_agg query in the "cancer_data.sql" to produce the data in json format.
<img src="https://github.com/CathyMatthee/predictor_app/blob/main/images/pgAdmin%204.png" alt="pgAdmin 4" width="300"/>

### Fetching and API Integration
1. From the root directory of the repo open app.py file
2. Install psycopg2 and numpy if you need to: use pip to install the libraries
3. Create a new file in the root directory called "config.py" which is where you provide your pgAdmin password in a safe manner. Add this text to the file: "password = "your_password_here" and replace "your_password_here" with your real password. 
    Save this in root directory of the cloned repo.
    This "config.py" file is referenced in the .gitignore file for safety reasons and is not present in the github repo. 
4. You can also add db_host = "localhost" to the file config.py if connecting to the local server.
    If you are connecting remotely to the database, you could potentially use the IP of one of the teammates servers by referencing their IP address instead of "localhost".
5. In git bash terminal activate your dev environment from the local repo and run "python app.py" to make a connection to the database wher the Flask app will serve the database data in JSON, dynamically to the machine learning models, ensuring they are trained on the most up-to-date data. 
6. Select the CTRL+click on the link that is output in the bash terminal that deploys the Flask locally in a window.
7. Select the "Predictor_App" option in the top navigation bar to go straight to the machine learning app that will predict cancer.
<img src="https://github.com/CathyMatthee/predictor_app/blob/main/images/Predictor%20App.png" alt="Predictor App" width="300"/>

# Machine Learning Model Builds

## Logistic Regression Model 1

### Overview
This part of the project focuses on building a logistic regression model to classify outcomes based on the dataset. The model was developed and trained to predict binary outcomes (such as B for Benign and M for Malignant in cancer diagnosis). Key steps included data preprocessing, training the model using default settings, and further improving performance through hyperparameter tuning with GridSearchCV.

Additionally, feature selection was attempted, followed by tuning and evaluation to compare the performance of the model with selected features against the one trained with all features. Here, we aimed to determine if the feature-selected model performed similarly for conversion into an app.

### Contents
- **Logistic_Regression_Model_1_ZR.ipynb**: This notebook contains the entire workflow of the logistic regression analysis, including data preprocessing, feature selection, model training, and evaluation results.
- **tuned_logistic_regression_model.pkl**: A saved version of the trained logistic regression model, which can be used for future predictions.

### Steps
1. **Data Preprocessing**: The dataset was cleaned by removing irrelevant columns, treating skewness of the data, and removing outliers. Categorical variables were encoded, and numerical features were scaled.
  
2. **Feature Selection**: Features were selected using Recursive Feature Elimination (RFE) to enhance model performance.

3. **Model Training**: The logistic regression model was trained using the scikit-learn library. Hyperparameter tuning was performed to optimize the model's performance using GridSearchCV.

### Model Evaluation
The model was evaluated using accuracy, precision, recall, F1 score, and confusion matrices using test data.

### Results
The model's performance slightly decreased after feature selection, which is reflected in:
- A small drop in accuracy (from 98% to 96%).
- Minor drops in precision and F1-scores for class 1 (malignant).
- A slight increase in misclassifications for class 0 (benign) from 1 to 2.

Therefore, feature selection with RFE slightly reduced the model's ability to differentiate between the classes. The full feature model performed better than the feature-selected model for logistic regression.

### Next Steps
We explored other machine learning algorithms such as Random Forest, SVM, or Neural Networks to compare performance.

## Support Vector Machine Model 1

## Random Forest Model 3

## Keras Tuner Model 4

# Looking Forward/Conclusion
1. Additional data will continue to promote model performance and the app could be used globally to collect data that doctors input into the app.
2. Use an ensemble approach using all the models where the ideal algorithm has low bias (accurately models the true relationshio in the data) and low variability (producing consistent predictions across different datasets).

### Sources
    - W3 Schools code used to build app navigation https://www.w3schools.com/bootstrap5/bootstrap_navs.php 
    - Chat gpt and Codepen used to convert README.md file into html and css formating for the web app pages including the predictor app.
    - Flask linking html pages: https://www.youtube.com/watch?v=VyICzbnf6q4
    - App runner web deployment: https://farzam.at/en/blog/deploy-flask-apps-aws-app-runner
