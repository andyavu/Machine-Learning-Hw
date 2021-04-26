#-------------------------------------------------------------------------
# AUTHOR        :   Andy Vu
# FILENAME      :   decision_tree.py
# SPECIFICATION :   Derive decision tree produced by the standard ID3 algorithm
# FOR           :   CS 4200 - Assignment #1
# TIME SPENT    :   02/17/2021 - 02/20/2021
#-----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv

db = []
X = []
Y = []

# reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: # skipping the header
            db.append (row)
            print(row)

# transfer the original training features to numbers and add to the 4D array X.
# For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]

# Age           : Young = 1, Prepresbyopic = 2, Presbyopic = 3
# Spectacle     : Myope = 1, Hypermetrope = 2
# Astigmatism   : Yes = 1, No = 2
# Tear Prod     : Reduced = 1, Normal = 2
for row in db:
    temp = []
    for i in range(4):
        if row[i] == "Young" or row[i] == "Myope" or row[i] == "Yes" or row[i] == "Reduced":
            temp.append(1)
        elif row[i] == "Prepresbyopic" or row[i] == "Hypermetrope" or row[i] == "No" or row[i] == "Normal":
            temp.append(2)
        else:
            temp.append(3)
    X.append(temp)

# transfer the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]

# Yes = 1, No = 2
for row in db:
    if row[4] == "Yes":
        Y.append(1)
    else:
        Y.append(2)

# fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

# plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()

