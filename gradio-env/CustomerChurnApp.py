import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter



# warnings.filterwarnings('ignore')

# Define your function that processes the input
def process_input(
    CustomerID, Gender, SeniorCitizens, Partner, Tenure, PhoneService, Multiplelines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
    StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
    MonthlyCharges, TotalCharges,Churn
):
    input_data = {
        "CustomerID": CustomerID,
        "Gender": Gender,
        "SeniorCitizens": SeniorCitizens,
        "Partner": Partner,
        "Tenure": Tenure,
        "PhoneService": PhoneService,
        "Multiplelines": Multiplelines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "Churn":Churn
    }

    # Perform some processing based on the inputs (you can define your logic here)
    final_data = pd.DataFrame([input_data])
    final_data.head(10)
    # result = final_data
    # return result
    #initialize combined data - final data, X test data
    train_data = final_data
    X_test = final_data
    #create a new feature YearlyCharges in each datasets
    train_data['ChargesperYear'] = train_data['MonthlyCharges']*12
    X_test['ChargesperYear'] = X_test['MonthlyCharges']*12

# Reorder the columns so that Churn Column to be the last one
    columns = ['CustomerID', 'Gender', 'SeniorCitizen', 'Partner', 'Dependents',
           'Tenure', 'PhoneService', 'MultipleLines', 'InternetService',
           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
           'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'ChargesperYear',
           'Churn']

    train_data = train_data.reindex(columns=columns)
    #Selecting only numerical columns and setting a named variable for both Train data and X_test
    num_train_data = train_data.select_dtypes(exclude="object")
    num_X_test = X_test.select_dtypes(exclude="object")
    
    #Dealing will categorical columns  on the train and X-test Sets
    cat_train_data = train_data.select_dtypes(include="object")
    #lets drop customer ID columns on trainset
    ct_train_data = cat_train_data.drop(columns=['CustomerID','Churn']) #= to match the columns in test data

    #for Test data
    cat_X_test = X_test.select_dtypes(include="object")
    #lets drop customer ID columns on trainset
    ct_X_test = cat_X_test.drop(columns=['CustomerID'])

    #ENCODING CATEGORICAL VALUES
    #Creating an instance of OneHotEncode
    enc= OneHotEncoder()
    # model to learn from Categorical X train data
    enc.fit(ct_train_data)
    # Transform the Categorical train data
    tct_train_data = pd.DataFrame(enc.transform(ct_train_data).toarray(), index=ct_train_data.index, columns=enc.get_feature_names_out(input_features=ct_train_data.columns))

# Transform the Categorical X_test data
    tct_X_test = pd.DataFrame(enc.transform(ct_X_test).toarray(), index=ct_X_test.index, columns=enc.get_feature_names_out(input_features=ct_X_test.columns))
    #Combining Categorical and number tranformed Xtrain and Xtest sets
    comb_train_data= pd.merge(left=num_train_data,right=tct_train_data, how='outer',left_index =True,right_index=True)
    comb_X_test= pd.merge(left=num_X_test,right=tct_X_test, how='outer',left_index =True,right_index=True)
    # Create StandardScaler instance
    std = StandardScaler()

    # Apply the scaler to the datasets
    final_train_data = pd.DataFrame(std.fit_transform(comb_train_data), columns=comb_train_data.columns, index=comb_train_data.index)

    # Use the same scaling parameters from the training data to scale the test data
    final_X_test = pd.DataFrame(std.transform(comb_X_test), columns=comb_X_test.columns, index=comb_X_test.index)
# Add the "CustomerID" and "Churn" columns back to final_train_data
    final_train_data["CustomerID"] = cat_train_data["CustomerID"]
    final_train_data["Churn"] = cat_train_data["Churn"]

    # Add the "CustomerID" column back to final_X_test
    final_X_test["CustomerID"] = cat_X_test["CustomerID"]

    final_train_data['Churn'] = final_train_data['Churn'].map({'Yes': 1, 'No': 0})

    # Define input components using gr.components

    #dataset to be split is:
    final_train_data
    # Separate features (X) and target (y) in combined data
    X = final_train_data.drop(columns=['Churn', 'CustomerID'])
    y = final_train_data['Churn']

    # Perform random shuffling
    X_shuffled = X.sample(frac=1, random_state=42)
    y_shuffled = y.iloc[X_shuffled.index]

    # Split into training (80%) and evaluation (20%) sets
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_shuffled)
    # Define SMOTE strategy
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    #Summarize class distribution before SMOTE
    print("Before SMOTE: ", Counter(y_train))

    #Apply SMOTE to balance the training set
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Summarize class distribution after SMOTE
    # print("After SMOTE: ", Counter(y_train_balanced))

    # Model training
    #initialize the model
    logreg = LogisticRegression()
    logreg.__str__()
    #using fit method to train
    logreg.fit(X_train_balanced, y_train_balanced)
    #predicting evaluation test y values using evaluation data
    X_eval_array = X_eval.values #==convert dataframe to numpy array
    y_eval_pred = logreg.predict(X_eval_array)
    #compute the metrics for classification report to check model accuracy
    reportlogreg = classification_report(y_eval, y_eval_pred, target_names=["Yes", "No"])

    logistic = logreg
 
    logistic.get_params() # To get the list of parameters we can manually tune

    parameters = {
    'C': [ 0.1 , 0.5 , 5, 7, 10],
    'max_iter': [200, 300, 500],
    'penalty': ['l2' ,  'l1'],
    'class_weight': 'Balanced',
    'class_weight': [None, 'balanced', {0: 1.0, 1: 2.0, 2: 0.5}]

    }

    searcher = GridSearchCV (logreg,
                            param_grid = parameters,
                            scoring = ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall'],
                            refit = 'balanced_accuracy',
                            cv= 5,
                            verbose = 3)
    searcher.fit(X_train_balanced,y_train_balanced)
    # Load the dataset
    X = X_train_balanced
    y = y_train_balanced

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature Selection
    k_best = SelectKBest(score_func=f_classif, k=2)
    X_selected = k_best.fit_transform(X_scaled, y)

    # Define the model 
    model = logreg

    # Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = cross_val_score(model, X_selected, y, cv=cv, scoring='accuracy')

    # Calculate the mean accuracy and standard deviation
    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)

#best_models = searcher.estimator
# print("Best Hyperparameters:", searcher.param_grid)
# print("Best Score:", searcher.scoring)
# Load the dataset

    X = X_train_balanced
    y = y_train_balanced

    # Feature Scaling (Standardization)
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Feature Selection
    k_best = SelectKBest(score_func=f_classif, k=2)
    X_selected = k_best.fit_transform(X_standardized, y)

    # Define and train the Logistic Regression model
    model = logreg
    model.fit(X_selected, y)

    # Make test predictions
    new_data_standardized = scaler.transform(X_train_balanced[:5])
    new_data_selected = k_best.transform(new_data_standardized)
    predicted_class = model.predict(new_data_selected)
    predicted_probabilities = model.predict_proba(new_data_selected)

    # print("Predicted Class:", predicted_class)
    # print("Predicted Probabilities:", predicted_probabilities)
    result = predicted_class
    return result
input_components = [
    gr.components.Textbox(label="CustomerID"),
    gr.components.Radio(label="Gender", choices=["Male", "Female"]),
    gr.components.Radio(label="SeniorCitizens", choices=["Yes", "No"]),
    gr.components.Radio(label="Partner", choices=["Yes", "No"]),
    gr.components.Number(label="Tenure"),
    gr.components.Radio(label="PhoneService", choices=["Yes", "No"]),
    gr.components.Dropdown(
        label="MultipleLines",
        choices=["Unknown", "No", "Yes"]
    ),
    gr.components.Dropdown(label="InternetService", choices=["DSL", "Fiber optic", "No"]),
    gr.components.Dropdown(label="OnlineSecurity", choices=["No", "Yes", "Unknown"]),
    gr.components.Dropdown(label="OnlineBackup", choices=["No", "Yes", "Unknown"]),
    gr.components.Dropdown(label="DeviceProtection", choices=["No", "Yes", "Unknown"]),
    gr.components.Dropdown(label="TechSupport", choices=["No", "Yes", "Unknown"]),
    gr.components.Dropdown(label="StreamingTV", choices=["No", "Yes", "Unknown"]),
    gr.components.Dropdown(label="StreamingMovies", choices=["No", "Yes", "Unknown"]),
    gr.components.Dropdown(label="Contract", choices=["Month-to-month", "One year", "Two year"]),
    gr.components.Radio(label="PaperlessBilling", choices=["Yes", "No"]),
    gr.components.Dropdown(label="PaymentMethod", choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
    gr.components.Number(label="MonthlyCharges"),
    gr.components.Number(label="TotalCharges")
]



# Create the Gradio interface

iface = gr.Interface(

    fn=process_input,
    inputs=input_components,
    outputs="number" , # You can specify the output type here.
    title = "Customer Churn prediction app"
)


# Launch the interface
iface.launch()
