### Project Proposal

# Goal of the Project:
The purpose for this project is to assist with breast cancer survial through early intervention. A website will be created for the use of doctors which will contain user inputted data based on the visual characteristics of a cancer. A machine learning model will be used to predict a diagnosis on whether the identified cancer is Benign or Malignant based on these visual characteristics. The features that will be used to assist in determining the diagnosis include: mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean compactness, mean concavity and mean concave points. The dataset identifed for this project is the Breast Cancer Wisconsin (Diagnostic) Data Set. 

# Dataset Link: 
https://www.kaggle.com/datasets/erdemtaha/cancer-data/data

# Screenshots:
![image](https://github.com/lhenry97/Group_1-Project_4/blob/main/Image.png)

# Method
The data will undergo cleaning and removal of any duplications or unnecessary data. Feature engineering will then be conducted on the dataset to ensure the data is in a usable state for machine learning. Postgres will be used to manage the database and the app.py flask app will be used to connect to the database to enable a website to call information from it. A number of machine learning models will be tested and evaluated on their prediction of the cancer diagnosis. The models include logistic regression, SVM, random forest and potentially Deep Neural Network. Each model will undergo optimisation and then they will be evaluated to select the best model. The selected machine learning model will then be used in the final website. The website will look similar to the above screenshot to enable a user to alter different visual characteristics and the model will output a predicted diagnosis.

# Licensing:
This Data has a CC BY-NC-SA 4.0 License.
https://creativecommons.org/licenses/by-nc-sa/4.0/

# Ethics:
This dataset contains a unique anonymous ID number for each patients cancer data. This is not considered a personally identifiable information as it is not linked back to any specific personal information of that patient such as a drivers license number or social security number.

# Workflow for Reproducibility

## Connecting to PostgreSQL with psycopg2 -->
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
![image of predictor app](Predictor App.png)