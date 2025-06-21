from django.shortcuts import render, redirect, HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate 
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from .forms import File_saving_form
from .models import File_saving_model
from django.contrib import messages
import pandas as pd
import pickle
import numpy as np
import os.path
from datetime import datetime

linear_reg_model = pickle.load(open("models/Linear_Regression.pkl", "rb"))
lasso_reg_model = pickle.load(open("models/Lasso_Regression.pkl", "rb"))
ridge_reg_model = pickle.load(open("models/Ridge_Regression.pkl", "rb"))
svr_reg_model = pickle.load(open("models/Support_Vector_Regression.pkl", "rb"))
decision_tree_reg_model = pickle.load(open("models/Decision_Tree_Regression.pkl", "rb"))
random_forest_reg_model = pickle.load(open("models/Random_Forest_Regression.pkl", "rb"))

# Create your views here.
def index(request):
    return render(request, "index.html")

def signup(request): 
    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]
        password = request.POST["password"]

        if User.objects.filter(username=username):
            messages.error(request, "Username Already Exist")
            return redirect('signup')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email Already Registered!!")
            return redirect('signup')

        my_user = User.objects.create_user(username, email, password)
        my_user.save()
        return redirect("login")

    return render(request, "signup.html")

def login(request):
    form = File_saving_form()
    if request.method == "POST":
        global username
        username = request.POST["username"]
        password = request.POST["password"]

        user_login = authenticate(username=username, password=password)
        if user_login is not None:
            auth_login(request, user_login)
            return render(request, "app.html", {'form':form})
        else:
            return render(request, "index.html")

    return render(request, "login.html")

def logout(request):
    auth_logout(request)
    return render(request, "index.html")

def predict_with_input_form(request):
    age = int(request.POST["age"]) 
    sex = int(request.POST["sex"])
    bmi = float(request.POST["bmi"])
    children = int(request.POST["children"])
    smoker = float(request.POST["smoker"])
    region = float(request.POST["region"]) 
    model_choice = float(request.POST["model_choice"])
    
    input_data = (age, sex,	bmi, children, smoker, region)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    if model_choice == 1:
        prediction = random_forest_reg_model.predict(input_data_reshaped)[0]
    elif model_choice == 2:
        prediction = decision_tree_reg_model.predict(input_data_reshaped)[0]
    elif model_choice == 3:
        prediction = svr_reg_model.predict(input_data_reshaped)[0]
    elif model_choice == 4:
        prediction = lasso_reg_model.predict(input_data_reshaped)[0]
    elif model_choice == 5:
        prediction = ridge_reg_model.predict(input_data_reshaped)[0]
    elif model_choice == 6:
        prediction = linear_reg_model.predict(input_data_reshaped)[0]
    
    prediction = prediction*82
    prediction = round(prediction, 2)

    # Saving prediction
    current_dateTime = datetime.now()
    csv_name = username+"_predictions.csv"
    csv_save_path = "the_app/static/saved_predictions/"+csv_name
    exist_already = os.path.isfile(csv_save_path)

    if sex == 0:
        sex = "male"
    elif sex == 1:
        sex = "female"

    if smoker == 0:
        smoker = "yes"
    elif smoker == 1:
        smoker = "no"

    if region == 0:
        region = "southeast"
    elif region == 1:
        region = "southwest"
    elif region == 2:
        region = "northeast"
    elif region == 3:
        region = "northwest"
    
    if model_choice == 1:
        model_choice = "Random Forest Regression"
    elif model_choice == 2:
        model_choice = "Decision Tree Regression"
    elif model_choice == 3:
        model_choice = "Support Vector Regression"
    elif model_choice == 4:
        model_choice = "Lasso Regression"
    elif model_choice == 5:
        model_choice = "Ridge Regression"
    elif model_choice == 6:
        model_choice = "Linear Regression"
    
    if exist_already == False:
        df = pd.DataFrame(columns=['Row No#', 'dateTime', 'age','sex','bmi','children','smoker','region', 'model_choice', 'prediction'])
        row_number = len(df)+1
        df.loc[len(df)] = [row_number, current_dateTime, age, sex, bmi, children, smoker, region, model_choice, prediction]
        df.to_csv(csv_save_path, index=False)
    else:
        df = pd.read_csv(csv_save_path)
        row_number = len(df)+1
        df.loc[len(df)] = [row_number, current_dateTime, age, sex, bmi, children, smoker, region, model_choice, prediction]
        df.to_csv(csv_save_path, index=False)

    csv_save_path_to_download = "static/saved_predictions/"+csv_name
    path_to_show_csv = "../"+csv_save_path_to_download

    return render(request, "display_preds.html", {"prediction":prediction, "csv_save_path_to_download":csv_save_path_to_download, "csv_name":csv_name, 'path_to_show_csv':path_to_show_csv})

def predict_with_file(request):
    if request.method == 'POST':
        form = File_saving_form(request.POST, request.FILES)
        if form.is_valid():
            myfile = form.cleaned_data['myfile']
            mymodel = File_saving_model(myfile=myfile)
            mymodel.save()
            df = pd.read_csv("the_app/static/uploaded_files/input_file.csv")
            # performing data preprocessing (same as performed on training data)
            df.replace({'sex':{'male':0,'female':1}}, inplace=True)
            df.replace({'smoker':{'yes':0,'no':1}}, inplace=True) 
            df.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)
            # performing prediction
            prediction = random_forest_reg_model.predict(df)[0]
            # adding prediction in the csv
            df["prediction"] = prediction
            # performing reverse data preprocessing (to save csv without encodings for categorical variables)
            df.replace({'sex':{0:'male', 1:'female'}}, inplace=True)
            df.replace({'smoker':{0:'yes', 1:'no'}}, inplace=True) 
            df.replace({'region':{0:'southeast', 1:'southwest', 2:'northeast', 3:'northwest'}}, inplace=True)
            # saving csv
            df.to_csv("the_app/static/saved_predictions/output_csv.csv", index=False)
            csv_save_path_to_download = "static/saved_predictions/output_csv.csv"
            path_to_show_csv = "../"+csv_save_path_to_download
            csv_name = "output_csv.csv"
            return render(request, "display_preds.html", {"prediction":prediction, "csv_save_path_to_download":csv_save_path_to_download, "csv_name":csv_name, 'path_to_show_csv':path_to_show_csv})
        else:
            return HttpResponse("Not Valid Input")
        
    return HttpResponse("Works")

def use_app(request):
    form = File_saving_form()
    if request.user.is_authenticated:
        return render(request, "app.html", {'form':form})
    else:
        return HttpResponse("Not logged In, Please Login to use the app!")