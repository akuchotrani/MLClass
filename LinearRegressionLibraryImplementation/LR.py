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
row = []
X_Matrix = []
Y_Price_Mat = []

X_Test = []
Y_Test = []

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
    global Y_Price_Mat
    Y_Price_Mat = np.matrix(Y_Price)

    ArrayBeds = np.array(x1_Beds)
    ArrayBaths = np.array(x2_Baths)
    ArraySqFeet = np.array(x3_SquareFeet)
    OneArray = np.ones((350,1))

    global X_Matrix
    X_Matrix =  np.column_stack((OneArray,ArrayBeds,ArrayBaths,ArraySqFeet))
    
    print("X_Matrix\n",X_Matrix,"\n")


def Linear_Regression():
    X_Matrix_Transpose = X_Matrix.transpose()
    print("X_Matrix Transpose\n",X_Matrix_Transpose,"\n")
    Inverse_Xtranspose_X = np.linalg.inv(np.matmul(X_Matrix_Transpose,X_Matrix))
    print("Inverse X_transpose_X:\n",Inverse_Xtranspose_X,"\n") 
    
    temp = np.squeeze(np.asarray(X_Matrix_Transpose))
    temp2 = np.squeeze(np.asarray(Y_Price_Mat))
    Xtranspose_Y = np.dot(temp,temp2)
    print("Xtranspose_Y:\n",Xtranspose_Y,"\n")


    global theta
    theta = np.matmul(Inverse_Xtranspose_X,Xtranspose_Y)
    print(theta)

def Plot_Result():
    # Visualising the Training set results
    #plt.plot(X_train, regressor.predict(X_train), color = 'blue')
    predictedPrices = np.zeros(350)
    print("theta 0 : ",theta[0])
    print("theta 1 : ",theta[1])
    print("theta 2 : ",theta[2])
    print("theta 3 : ",theta[3])
    for x in range(0,350):
        predictedPrices[x] = theta[0] + theta[1]*x1_Beds[x] + theta[2]*x2_Baths[x] + theta[3]*x3_SquareFeet[x]
    
    print(predictedPrices)
    plt.title('Squarefeet vs Price (Training set)')
    plt.xlabel('Squre Feet')
    plt.ylabel('Price of houses')
    plt.scatter(x3_SquareFeet,predictedPrices, color = 'red')
    
    line = np.zeros(350)
    for i in range (0,350):
        line[i] = theta[3]*x3_SquareFeet[i] + theta[0]
        
    print(line)
    
    plt.show()


def main():
    Read_Data()
    Split_Data()
    Create_Matrix()
    Linear_Regression()
    Plot_Result()


if __name__ == "__main__":
    main()