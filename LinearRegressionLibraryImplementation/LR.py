# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:46:44 2018

@author: aakash.chotrani
"""

import csv
import numpy as np

CSV_Data = csv.reader(open('redfinBothell.csv', newline=''))
Y_Price = []
x1_Beds = []
x2_Baths = []
x3_SquareFeet = []
row = []

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


def Create_Matrix():
    Y_Price_Mat = np.matrix(Y_Price)
#    MatrixRows = []
#    for i in range(0,350):
#        MatrixRows.append(x1_Beds[i]+","+x2_Baths[i]+","+x3_SquareFeet[i]+";")
        
    MatrixBeds = np.array(x1_Beds)
    MatrixBaths = np.array(x2_Baths)
    
    np.reshape(MatrixBeds,(,350))
    
    #MatrixSquareFeet = np.array(x3_SquareFeet)
    
    #tempMatrix = np.concatenate((MatrixBeds,MatrixBaths.T),axis = 1)
    #print(tempMatrix.shape)
    
def main():
    Read_Data()
    Create_Matrix()


if __name__ == "__main__":
    main()