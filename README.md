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

