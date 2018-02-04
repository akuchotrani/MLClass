# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:46:44 2018

@author: aakash.chotrani
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

CSV_Data = csv.reader(open('redfinBothell.csv', newline=''))
Y_Price = []
x1_Beds = []
x2_Baths = []
x3_SquareFeet = []
X_Matrix = []


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


def Read_Data1():
    isHeading = True
    
    for row in CSV_Data:
        #skip the first row of csv file
        if isHeading == True:
            isHeading = False
            continue
    
        if(row[7] == ""):
            Y_Price.append("0")
        else:
            Y_Price.append((row[7]))
    
        if(row[8] == ""):
            x1_Beds.append("0")
        else:
            x1_Beds.append((row[8]))
    
        if(row[9] == ""):
            x2_Baths.append("0")
        else:
            x2_Baths.append(row[9])
            
        if(row[11] == ""):
            x3_SquareFeet.append("0")
        else:
            x3_SquareFeet.append((row[11]))


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
    
    


def main():
    Read_Data()
    Split_Data()
    Create_Matrix()
    Linear_Regression()
    Plot_Result()


if __name__ == "__main__":
    main()