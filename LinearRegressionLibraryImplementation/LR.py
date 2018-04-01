# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:46:44 2018

@author: aakash.chotrani
"""

import csv
import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


CSV_Data = csv.reader(open('redfinBothell.csv', newline=''))
Y_Price = []
x1_Beds = []
x2_Baths = []
x3_SquareFeet = []
X_Matrix = []

MAX_ITERATIONS = 200
LEARNING_RATE = 0.05


def Read_Data():
    isHeading = True
    
    for row in CSV_Data:
        #skip the first row of csv file
        if isHeading == True:
            isHeading = False
            continue
    
        if(row[7] == ""):
            Y_Price.append(0)
        else:
            Y_Price.append(int(row[7]))
    
        if(row[8] == ""):
            x1_Beds.append(0)
        else:
            x1_Beds.append(float(row[8]))
    
        if(row[9] == ""):
            x2_Baths.append(0)
        else:
            x2_Baths.append(float(row[9]))
            
        if(row[11] == ""):
            x3_SquareFeet.append(0)
        else:
            x3_SquareFeet.append(int(row[11]))
            
def Clean_Data_Average():
    global Y_Price
    global x1_Beds
    global x2_Baths
    global x3_SquareFeet
    
    
    #Cleaning Price with average price
    averagePrice = sum(Y_Price)/len(Y_Price)
    
    for x in range(0,len(Y_Price)):
        if(Y_Price[x] == 0):
            Y_Price[x] = averagePrice
            
            
    #Cleaning X1_Beds with average price
    averageBeds = sum(x1_Beds)/len(x1_Beds)
    #print(averageBeds)
    for x in range(0,len(x1_Beds)):
        if(x1_Beds[x] == 0):
            x1_Beds[x] = averageBeds
    
    #Cleaning x2_Baths with average price
    averageBaths = sum(x2_Baths)/len(x2_Baths)
    #print(averageBaths)
    for x in range(0,len(x2_Baths)):
        if(x2_Baths[x] == 0):
            x2_Baths[x] = averageBaths
    
    #Cleaning x2_Baths with average price
    averagePriceSqFeet = sum(x3_SquareFeet)/len(x3_SquareFeet)
    #print(averagePriceSqFeet)
    for x in range(0,len(x3_SquareFeet)):
        if(x3_SquareFeet[x] == 0):
            x3_SquareFeet[x] = averagePriceSqFeet
            
            
def Clean_Data_Median():
    global Y_Price
    global x1_Beds
    global x2_Baths
    global x3_SquareFeet
    
    
    #Cleaning Price with average price
    medianPrice = statistics.median(Y_Price)
    for x in range(0,len(Y_Price)):
        if(Y_Price[x] == 0):
            Y_Price[x] = medianPrice
            
            
    #Cleaning X1_Beds with average price
    medianBeds = statistics.median(x1_Beds)
    #print(medianPrice)
    for x in range(0,len(x1_Beds)):
        if(x1_Beds[x] == 0):
            x1_Beds[x] = medianBeds
    
    #Cleaning x2_Baths with average price
    medianBaths = statistics.median(x2_Baths)
    #print(medianBaths)
    for x in range(0,len(x2_Baths)):
        if(x2_Baths[x] == 0):
            x2_Baths[x] = medianBaths
    
    #Cleaning x2_Baths with average price
    medianSqFeeet = statistics.median(x3_SquareFeet)
    #print(medianSqFeeet)
    for x in range(0,len(x3_SquareFeet)):
        if(x3_SquareFeet[x] == 0):
            x3_SquareFeet[x] = medianSqFeeet
    


def Split_Data():
    global train_x1_Beds
    train_x1_Beds = x1_Beds[0:250]
    global test_x1_Beds
    test_x1_Beds = x1_Beds[-100:]    
    
    global train_x2_Baths
    train_x2_Baths = x2_Baths[0:250]
    global test_x2_Baths
    test_x2_Baths = x2_Baths[-100:]
    
    global train_x3_SqFeet
    train_x3_SqFeet = x3_SquareFeet[0:250]
    global test_x3_SqFeet
    test_x3_SqFeet = x3_SquareFeet[-100:]
    
    global train_Y_Price
    train_Y_Price = Y_Price[0:250]
    global test_Y_Price
    test_Y_Price = Y_Price[-100:]

def Create_Matrix():

    ArrayBeds = np.array(train_x1_Beds)
    ArrayBaths = np.array(train_x2_Baths)
    ArraySqFeet = np.array(train_x3_SqFeet)
    OneArray = np.ones((250,1))

    global X_Matrix
    X_Matrix =  np.column_stack((OneArray,ArrayBeds,ArrayBaths,ArraySqFeet))
    
    #print("X_Matrix\n",X_Matrix,"\n")


def Linear_Regression():
    X_Matrix_Transpose = X_Matrix.transpose()
    #print("X_Matrix Transpose\n",X_Matrix_Transpose,"\n")
    Inverse_Xtranspose_X = np.linalg.inv(np.matmul(X_Matrix_Transpose,X_Matrix))
    #print("Inverse X_transpose_X:\n",Inverse_Xtranspose_X,"\n") 
    
    temp = np.squeeze(np.asarray(X_Matrix_Transpose))
    temp2 = np.squeeze(np.asarray(train_Y_Price))
    Xtranspose_Y = np.dot(temp,temp2)
    #print("Xtranspose_Y:\n",Xtranspose_Y,"\n")


    global theta
    theta = np.matmul(Inverse_Xtranspose_X,Xtranspose_Y)
    #print(theta)
    
    
    
def Linear_Regression_Gradient_Descent():
    X_Matrix_Transpose = X_Matrix.transpose()
    m = 4 #X_Matrix shape
    #print(X_Matrix_Transpose)
    WEIGHTS = np.ones(4)
    print(WEIGHTS)
    for iteration in range(0,MAX_ITERATIONS):
        hypothesis = np.dot(X_Matrix,WEIGHTS)
        #print(hypothesis)
        
        loss = hypothesis - train_Y_Price
        cost = np.sum(loss ** 2)/(2*m)
        
        gradient = np.dot(X_Matrix_Transpose,loss)
        print("iteration: ",iteration," cost: ",cost," Gradient: ",gradient)
        WEIGHTS = WEIGHTS - gradient.dot(LEARNING_RATE)
    
    print("Weights:",WEIGHTS)
    return WEIGHTS


def Plot_Result():
    # Visualising the Training set results
    #print("theta 0 : ",theta[0])
    #print("theta 1 : ",theta[1])
    #print("theta 2 : ",theta[2])
    #print("theta 3 : ",theta[3])
    predictedPricesTrain = np.zeros(250)

    for x in range(0,250):
        predictedPricesTrain[x] = theta[0] + theta[1]*train_x1_Beds[x] + theta[2]*train_x2_Baths[x] + theta[3]*train_x3_SqFeet[x]
    
    #print(predictedPrices)
    plt.title('Squarefeet vs Price (Training set)')
    plt.xlabel('Squre Feet')
    plt.ylabel('Price of houses')
    plt.scatter(train_x3_SqFeet,predictedPricesTrain, color = 'red')
    plt.scatter(train_x3_SqFeet,train_Y_Price,color = 'orange')
    plt.show()
    
    predictedPricesTest = np.zeros(100)
    for x in range(0,100):
        predictedPricesTest[x] = theta[0] + theta[1]*test_x1_Beds[x] + theta[2]*test_x2_Baths[x] + theta[3]*test_x3_SqFeet[x]
    plt.title('Squarefeet vs Price (Test set)')
    plt.xlabel('Squre Feet')
    plt.ylabel('Price of houses')
    plt.scatter(test_x3_SqFeet,predictedPricesTest, color = 'green')
    plt.scatter(test_x3_SqFeet,test_Y_Price,color = 'blue')
    plt.show()
    


def PerformFeatureScaling():
    global X_Matrix
    sc_X = StandardScaler()
    X_Matrix = sc_X.fit_transform(X_Matrix)


def main():
    Read_Data()
    Clean_Data_Average()
    #Clean_Data_Median()
    Split_Data()
    Create_Matrix()
    Linear_Regression()
    PerformFeatureScaling()
    #Linear_Regression_Gradient_Descent()
    global theta
    #theta = Linear_Regression_Gradient_Descent()
    Plot_Result()


if __name__ == "__main__":
    main()