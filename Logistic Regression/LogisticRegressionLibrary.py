# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 13:10:15 2018

@author: aakash.chotrani
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math

################################################################################
################################################################################

MAX_ITERATIONS = 100
LEARNING_RATE = 0.001
Y_Admit = []
x1_gre = []
x2_gpa = []
x3_rank = []

theta0 = 0.5
theta1 = 0.5
theta2 = 0.5
theta3 = 0.5

################################################################################
################################################################################

def Read_Data():
    
    CSV_Data = csv.reader(open('Admit.csv', newline=''))
    isHeading = True
    
    for row in CSV_Data:
        #skip the first row of csv file
        if isHeading == True:
            isHeading = False
            continue
    
        Y_Admit.append(float(row[0]))
        x1_gre.append(float(row[1]))
        x2_gpa.append(float(row[2]))
        x3_rank.append(float(row[3]))
        

################################################################################
################################################################################

def CreateMatrix():
    
    Array_Ones = np.ones((400,1))
    Array_Gre = np.array(x1_gre)
    Array_Gpa = np.array(x2_gpa)
    Array_Rank = np.array(x3_rank)

    global X_Matrix
    X_Matrix =  np.column_stack((Array_Ones,Array_Gre,Array_Gpa,Array_Rank))
    
    global Theta_Matrix
    Theta_Matrix = np.column_stack((theta0,theta1,theta2,theta3))

################################################################################
################################################################################
    
def PerformFeatureScaling():
    global X_Matrix
    sc_X = StandardScaler()
    X_Matrix = sc_X.fit_transform(X_Matrix)
    

def Spit_Dataset():
    
    global X_Matrix_Train
    X_Matrix_Train = X_Matrix[0:300]
    
    global X_Matrix_Test
    X_Matrix_Test = X_Matrix[300:400]
    
    global Y_Admit_Train
    Y_Admit_Train = Y_Admit[0:300]

    global Y_Admit_Test
    Y_Admit_Test = Y_Admit[300:400]

################################################################################
################################################################################

def Logistic_Regression_Gradient_Descent():
    
    global Theta_Matrix
    
    for iteration in range(0,MAX_ITERATIONS):
        #print("Iteration: ",iteration,"Theta Matrix: ",Theta_Matrix)
        for matrix_iteration in range(0,len(X_Matrix_Train)):
            #print(X_Matrix_Train[matrix_iteration])
            result = np.dot(X_Matrix_Train[matrix_iteration],Theta_Matrix[0])
            print("Theta[0]",Theta_Matrix[0])
            
            if result > 0:
                sigmaFunction = 1/(1+math.exp(-result))
            else:
                sigmaFunction = math.exp(result)/(math.exp(result) + 1)
            
            tempGradient = (Y_Admit_Train[matrix_iteration] - sigmaFunction)
            Gradient = np.dot(tempGradient,X_Matrix_Train[matrix_iteration])
            Gradient = Gradient * LEARNING_RATE
            Theta_Matrix = np.subtract(Theta_Matrix,Gradient)

    
################################################################################
################################################################################
            
def Predict_Results():

    Y_Prediction = []
    for item in range(0,len(Y_Admit_Test)):
        result = np.dot(Theta_Matrix,X_Matrix_Test[item])
        if result > 0:
            Y_Prediction.append(1)
        else:
            Y_Prediction.append(0)


    correct_counter = 0
    incorrect_counter = 0
    for item in range(0,len(Y_Admit_Test)):
        if(Y_Admit_Test[item] == Y_Prediction[item]):
            correct_counter = correct_counter + 1
        else:
            incorrect_counter = incorrect_counter + 1
        
    print("Correct Predictions: ",correct_counter)
    print("InCorrect Predictions: ",incorrect_counter)

        


def main():
    Read_Data()
    CreateMatrix()
    PerformFeatureScaling()
    Spit_Dataset()
    Logistic_Regression_Gradient_Descent()
    Predict_Results()

if __name__ == "__main__":
    main()