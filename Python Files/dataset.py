# -*- coding: utf-8 -*-
"""
This script generates data set from different optimization algorithms and random sampling
Author: Zuardin Akbar

Built based on process_logs_hypervolume.py by Thomas Wortmann
"""

import os
import pandas as pd
import numpy as np
from  matplotlib import pyplot as plt

# Input Variables

# Filepath to the folder with the logs.
# The results will be written here as well.
# The logs should be the only text files in that folder.

filePath = "C:\\Users\\harri\\OneDrive\\Documents\\ITECH2020\\Academic Materials\\21_Som Sem\\3064044_Computing in Architecture\\2_Assignments\\_Final\\_Final Logs\\MOO\\SupML\\"


# Log Name Settings (used to parse the solver from the log's file name)
log_name_limit = "_" # With what symbol should the file name be parsed?
log_name_solver = 2 # What part of the file name is the solver?

parameters_count = 90
objectives_count = 2
parameters_columns = []
objective_columns = []

#Functions
def parseObjs(filePath):
    '''Parse all objective from all logs in the file path'''
    objectivesList = []
    
    #Get text files 
    textFiles = []

    for file in os.listdir(filePath):
    	if file.endswith(".txt"):
    		textFiles.append(file)
    
    print (len(textFiles))
    
    #Read in all objectives
    for i, fileName in enumerate(textFiles):
        #Reset counter
        count = 0
        
        with open(filePath + fileName, "r") as file:
               for line in file:
                   words = line.split(" ")
                   objectiveStrs = words[5].split(",")   
                   objectives = list(map(lambda obj: float(obj), objectiveStrs))
                   objectivesList.append(objectives)
                   count += 1
                   
    print(f"{i + 1}/{len(textFiles)} text files parsed")
    print(f"Parsed {len(objectivesList)} sets of {len(objectivesList[0])} objectives.")
    assert i + 1 == len(textFiles)
    
    return objectivesList
                      
def splitOpossumLine(line, log_limit, log_time, log_parameters, log_objectives):
    words = line.split(log_limit)

    #Read objectives
    objectiveStrs = words[log_objectives].split(",")                
    objectives = list(map(lambda obj: float(obj), objectiveStrs))

    #Read parameters
    parameterStr = words[log_parameters].split(",")
    parameters = list(map(lambda obj: float(obj), parameterStr))
            
    #Read time
    times = words[log_time].split(":")
    #Process Time
    hrs = int(float(times[0]))
    min = int(float(times[1]))
    sec = int(float(times[2][:-1]))

    return(objectives, parameters, hrs, min, sec)

def processLog(filePath, fileName):
    parameterList = []
    objectivesList = []
    


    with open(filePath + fileName, "r") as file:
        for line in file:
            
            #Opossum Logs Settings	
            objectives, parameters, hrs, min, sec = splitOpossumLine(line, log_limit = " ", log_time = 1,  log_parameters = 3, log_objectives = 5)
            parameterList.append(parameters)
            objectivesList.append(objectives)      
            
            global parameters_count
            global objectives_count
            
            parameters_count = len(parameters)
            objectives_count = len(objectives)
            
        print (fileName, parameters_count, objectives_count)

    return [fileName, objectivesList, parameterList]

#Main
if __name__ == "__main__":
    prevSolverName = ''
    logLines = []
    multiObjs = []
    parameters = []
    solvers = []
    
    #Get Objective List
    print("\nParsing objectives ...")
    objectivesList = parseObjs(filePath)

    #Get text files 
    textFiles = []
    
    for file in os.listdir(filePath):
    	if file.endswith(".txt"):
    		textFiles.append(file)
    
     
    #Process text files
    for file in textFiles:
        solverName = (file.split(log_name_limit))[log_name_solver]
        logValues = processLog(filePath, file)
        
        # New solver    
        if prevSolverName != solverName:        
           
            # Record new solver name
            solvers.append(solverName)
            prevSolverName = solverName     
          
            
        #Store multi-objective values for best-know front calculation
        multiObjs.extend(logValues[1])
        parameters.extend(logValues[2])
    
    #Generate csv datasets
    parameters_columns = ["P_" + str(i) for i in range(parameters_count)]
    objectives_columns = ["O_" + str(i) for i in range(objectives_count)]
    
    parameters_columns.extend(objectives_columns)
    
    combined_list = np.concatenate((parameters, multiObjs), axis=1)
    
    dataset = pd.DataFrame(combined_list)
    dataset.columns = parameters_columns
    
    
    dataset.to_csv(filePath + "dataset.csv",
                         index = False)
    
    
    #Plot data to check outliers
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(dataset['P_0'], dataset['O_0'])
    ax.set_xlabel('Parameter 0')
    ax.set_ylabel('Objective 0')
    plt.show()
   
    