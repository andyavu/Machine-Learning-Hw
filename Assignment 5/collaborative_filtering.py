#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4200- Assignment #5
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('trip_advisor_data.csv', sep=',', header=0) #reading the data by using the Pandas library ()

#iterate over the other 99 users to calculate their similarity with the active user (user 100) according to their category ratings (user-item approach)
   # do this to calculate the similarity:
   #vec1 = np.array([[1,1,0,1,1]])
   #vec2 = np.array([[0,1,0,1,1]])
   #cosine_similarity(vec1, vec2)
   #do not forget to discard the first column (User ID) when calculating the similarities
   #--> add your Python code here

   #find the top 10 similar users to the active user according to the similarity calculated before
   #--> add your Python code here

   #Compute a prediction from a weighted combination of selected neighbors’ for both categories evaluated (galleries and restaurants)
   #--> add your Python code here



