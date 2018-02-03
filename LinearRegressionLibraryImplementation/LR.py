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
X_Matrix = []
Y_Price_Mat = []

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
    global Y_Price_Mat
    Y_Price_Mat = np.matrix(Y_Price)

        
    ArrayBeds = np.array(x1_Beds)
    ArrayBaths = np.array(x2_Baths)
    OneArray = np.ones((350,1))
    
    global X_Matrix
    X_Matrix =  np.column_stack((OneArray,ArrayBeds,ArrayBaths))
    
    print(X_Matrix)

    
def main():
    Read_Data()
    Create_Matrix()


if __name__ == "__main__":
    main()