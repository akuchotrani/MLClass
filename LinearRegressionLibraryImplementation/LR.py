# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:46:44 2018

@author: aakash.chotrani
"""

import csv

CSV_Data = csv.reader(open('redfinBothell.csv', newline=''))
Y_Price = []
x1_Beds = []
x2_Baths = []
x3_SquareFeet = []
    
def Read_Data():
    isHeading = True
    
    for row in CSV_Data:
        
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
            x1_Beds.append(int(row[8]))
    
        if(row[9] == ""):
            x2_Baths.append(0)
        else:
            x2_Baths.append(row[9])
            
        if(row[11] == ""):
            x3_SquareFeet.append(0)
        else:
            x3_SquareFeet.append(int(row[11]))
    
    
def main():
    Read_Data()


if __name__ == "__main__":
    main()