#-------------------------------------------------------------------------
# AUTHOR        :   Andy Vu
# FILENAME      :   find_s.py
# SPECIFICATION :   Reads dataset from contact_lens.csv and performs Find-S algorithm
# FOR           :   CS 4200 - Assignment #1
# TIME SPENT    :   02/17/2021 - 02/20/2021
#-----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
import csv

num_attributes = 4
db = []
print("\n The Given Training Data Set \n")

# reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: # skipping the header
            db.append(row)
            print(row)

print("\n The initial value of hypothesis: ")
hypothesis = ['0'] * num_attributes # representing the most specific possible hypothesis
print(hypothesis)

# find the first positive training data in db and assign it to the vector hypothesis
first_pos = 0
for row in db:
    if row[4] == "Yes":
        for i in range(len(row) - 1):
            hypothesis[i] = row[i]
        break
    first_pos += 1

# find the maximally specific hypothesis according to your training data in db and assign it to the vector hypothesis (special characters allowed: "0" and "?")
for row in range(first_pos + 1, len(db)):
    if db[row][4] == "Yes":
        for i in range(4):
            if db[row][i] != hypothesis[i]:
                hypothesis[i] = "?"

print("\n The Maximally Specific Hypothesis for the given training examples found by Find-S algorithm:")
print(hypothesis)
