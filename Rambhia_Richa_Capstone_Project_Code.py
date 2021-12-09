"""
Capstone Project

Copyright (c) 2021 
Licensed 
Written by Richa Rambhia

"""


#pip install pandas
import pandas as pd

#pip install numpy
import numpy as np

#pip install os_sys
from sys import exit

#pip install matplotlib
import matplotlib.pyplot as plt

#pip install sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score



# Data Extraction:

#reading csv file
def csv_read():
    try:
        with open("D:/ALY 6140/Capstone Project Dataset/SPI Data.csv") as csv_file:
            csv_data = pd.read_csv(csv_file)
            csv_file.close()
            
        return csv_data
    
    except FileNotFoundError as error_msg:
        print(error_msg)




# Data Cleanup:

#renaming column names
def rename_column(csv_data):
    csv_data.rename(columns = {'ï»¿country':'Country', 'date':'Date', 'region':'Region'}, inplace = True)
    
    
#filling in missing values
def missing_values(csv_data):
    csv_data['Date'].fillna(value = 0, inplace=True)
    csv_data['SPI.INDEX.PIL1 Pillar 1  - Data Use - Score'].fillna(value = 0, inplace=True)
    csv_data['SPI.INDEX.PIL2 Pillar 2 - Data Services - Score'].fillna(value = 0, inplace=True)
    csv_data['SPI.INDEX.PIL3 Pillar 3 - Data Products - Score'].fillna(value = 0, inplace=True)
    csv_data['SPI.INDEX.PIL4 Pillar 4 - Data Sources - Score'].fillna(value = 0, inplace=True)
    csv_data['SPI.INDEX.PIL5 Pillar 5 - Data Infrastructure - Score'].fillna(value = 0, inplace=True)
    csv_data['SPI.INDEX SPI Overall Score'].fillna(value = 0, inplace=True)
    
    
    
# Data Visualization:
    
#graph 1
def data_services_graph(csv_data):

    plt.plot(csv_data['Date'],csv_data['SPI.INDEX.PIL2 Pillar 2 - Data Services - Score'])   
    plt.xlabel("Date") 
    plt.ylabel("Data Services Score")
    plt.title("SPI DATA ANALYSIS")
    plt.show()

#graph 2
def data_product_graph(csv_data):

    plt.bar(csv_data['Date'],csv_data['SPI.INDEX.PIL3 Pillar 3 - Data Products - Score'])   
    plt.xlabel("Date") 
    plt.ylabel("Data Products Score")
    xticks = np.arange(2004,2020,2)
    plt.xticks(xticks)
    yticks = np.arange(0,100,10)
    plt.yticks(yticks)
    plt.title("SPI DATA ANALYSIS")
    plt.show()
    
#graph 3
def overall_score_graph(csv_data):
    plt.bar(csv_data['Region'],csv_data['SPI.INDEX SPI Overall Score'])   
    plt.xlabel("Region") 
    plt.ylabel("Overall Score")
    plt.title("SPI DATA ANALYSIS")
    yticks = np.arange(0,100,10)
    plt.yticks(yticks)
    plt.xticks(rotation=90, ha="right")
    plt.show()
    
#graph 4
def services_graph(csv_data):
    plt.scatter(csv_data['SPI.INDEX.PIL2 Pillar 2 - Data Services - Score'],csv_data['Region'],color='orange')
    plt.xlabel("Data Services Score")
    plt.ylabel("Region")
    xticks = np.arange(0,100,10)
    plt.xticks(xticks)
    plt.title("SPI DATA ANALYSIS")
    plt.show()

#graph 5
def products_graph(csv_data):
    plt.scatter(csv_data['SPI.INDEX.PIL3 Pillar 3 - Data Products - Score'],csv_data['Region'],marker="x",color='lightblue')
    plt.xlabel("Data Products Score")
    plt.ylabel("Region")
    xticks = np.arange(0,100,10)
    plt.xticks(xticks)
    plt.title("SPI DATA ANALYSIS")
    plt.show()



# Descriptive Analytics:
    
#describing the data
def describe_data(csv_data):
    print(csv_data.info())
    
    print("\nDescribing the data:\n",csv_data.describe(include='all'))
    
    print("\nDisplaying starting records of dataset:\n",csv_data.head())
    print("\nDisplaying end records of dataset:\n",csv_data.tail())
    
    print("\nDisplaying the type of each column of dataset:\n",csv_data.dtypes)
    
    print("\nDisplaying the minimum value of Data Services score: ",csv_data['SPI.INDEX.PIL2 Pillar 2 - Data Services - Score'].min())
    print("\nDisplaying the maximum value of Data Services score: ",csv_data['SPI.INDEX.PIL2 Pillar 2 - Data Services - Score'].max())
    print("Displaying the mean of Data Services score:",csv_data['SPI.INDEX.PIL2 Pillar 2 - Data Services - Score'].mean())
    print("\nDisplaying the minimum value of Data Products score: ",csv_data['SPI.INDEX.PIL3 Pillar 3 - Data Products - Score'].min())
    print("\nDisplaying the maximum value of Data Products score: ",csv_data['SPI.INDEX.PIL3 Pillar 3 - Data Products - Score'].max())
    print("Displaying the mean of Data Products score:",csv_data['SPI.INDEX.PIL3 Pillar 3 - Data Products - Score'].mean())
    print("\nDisplaying the minimum value of SPI overall score: ",csv_data['SPI.INDEX SPI Overall Score'].min())
    print("\nDisplaying the maximum value of SPI overall score: ",csv_data['SPI.INDEX SPI Overall Score'].max())     
    print("\nDisplaying the count of SPI overall score: ",csv_data['SPI.INDEX SPI Overall Score'].count())
    print("Displaying the mean of SPI overall score:",csv_data['SPI.INDEX SPI Overall Score'].mean())
    
     


# Predictive Analytics:

def model(csv_data):  

    #converting object to categorical value and labeling    
    csv_data["Region"] = csv_data["Region"].astype('category')   
    csv_data["Region"] = csv_data["Region"].cat.codes
    
    csv_data["Country"] = csv_data["Country"].astype('category')
    csv_data["Country"] = csv_data["Country"].cat.codes
    
    
    x = csv_data[['Date']]
    y = csv_data[['SPI.INDEX SPI Overall Score']]

    #linear regression model
    print("\nLinear Regression Model")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    regressor = LinearRegression()    
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    
    cutoff = 0.5                           
    y_pred_classes = np.zeros_like(y_pred)    
    y_pred_classes[y_pred > cutoff] = 1 
    
    y_test_classes = np.zeros_like(y_pred)
    y_test_classes[y_test > cutoff] = 1    

    print("\nConfusion Matrix is:\n")
    print(confusion_matrix(y_test_classes, y_pred_classes))
    print("\nClassification Report is:\n")
    print(classification_report(y_test_classes, y_pred_classes))
    acc = accuracy_score(y_true=y_test_classes, y_pred=y_pred_classes)*100
    print("\nAccuracy is:",acc)
    


#main
if __name__=="__main__":

    #data extraction
    csv_data = csv_read()
    
    #data cleaning
    rename_column(csv_data)
    missing_values(csv_data)
    
    
    while(True):
        user_input = input("\nSelect from the options below:\n 1. Data Visualization\n 2. Descriptive Analysis\n 3. Predictive Analysis\n 4. Exit\n")
        
        if user_input == '1':
            #data visualization
            while(True):
                graph_input = input("\nSelect from the options below for viewing the graphs:\n 1. SPI Data Services Graph\n 2. SPI Data Products Graph\n 3. SPI Overall Score Graph\n 4. Region vs SPI Data Services Graph\n 5. Region vs SPI Data Products Graph\n 6. Exit\n")
                if graph_input == '1':
                    data_services_graph(csv_data)
                    
                elif graph_input == '2':
                    data_product_graph(csv_data)
                    
                elif graph_input == '3':
                    overall_score_graph(csv_data)
                    
                elif graph_input == '4':
                    services_graph(csv_data)
                    
                elif graph_input == '5':
                    products_graph(csv_data)
                    
                else:
                    break
                    
        elif user_input == '2':
            #descriptive analysis
            describe_data(csv_data)
    
        elif user_input == '3':
            #predictive analysis
            model(csv_data)
            
        else:
            exit()
    
    
    