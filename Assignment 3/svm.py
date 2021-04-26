#-------------------------------------------------------------------------
# AUTHOR        : Andy Vu
# FILENAME      : svm.py
# SPECIFICATION : Simulates a grid search, trying to find which combination
#                 of four SVM hyperparameters (C, degree, kernel, and decision_function_shape)
#                 leads you to the best prediction performance.
# FOR           : CS 4210 - Assignment #3
# TIME SPENT    : 04/04/2021-04/07/2021
#-------------------------------------------------------------------------

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn import svm
import csv

dbTraining = []
dbTest = []
X_training = []
Y_training = []
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highestAccuracy = 0

# reading the data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
    reader = csv.reader(trainingFile)
    for i, row in enumerate(reader):
        X_training.append(row[:-1])
        Y_training.append(row[-1])

# reading the data in a csv file
with open('optdigits.tes', 'r') as testingFile:
    reader = csv.reader(testingFile)
    for i, row in enumerate(reader):
        dbTest.append(row)

c_val = 0
d_val = 0
k_val = ""
w_val = ""
# created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
for i in c: # iterates over c
    for d in degree: # iterates over degree
        for k in kernel: # iterates kernel
            for w in decision_function_shape: # iterates over decision_function_shape
                # Create an SVM classifier that will test all combinations of C, degree, kernel, and decision_function_shape as hyperparameters. For instance svm.SVC(C=1)
                clf = svm.SVC(C=i, degree=d, kernel=k, decision_function_shape=w)

                # Fit Random Forest to the training data
                clf.fit(X_training, Y_training)

                accuracy = 0
                # make the classifier prediction for each test sample and start computing its accuracy
                for testSample in dbTest:
                    class_predicted = clf.predict([testSample[:-1]])
                    if int(class_predicted[0]) == int(testSample[-1]):
                        accuracy += 1
                accuracy /= len(dbTest)

                # check if the calculated accuracy is higher than the previously one calculated. If so, update update the highest accuracy and print it together with the SVM hyperparameters
                # Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                if accuracy > highestAccuracy:
                    highestAccuracy = accuracy
                    c_val = i
                    d_val = d
                    k_val = k
                    w_val = w
                    print("Highest SVM accuracy so far: " + str(highestAccuracy) + ", Parameters: C=" + str(i) + ", degree=" + str(d) + ", kernel=" + str(k) + ", decision_function_shape=" + str(w))

# print the final, highest accuracy found together with the SVM hyperparameters
# Example: "Highest SVM accuracy: 0.95, Parameters: a=10, degree=3, kernel= poly, decision_function_shape = 'ovr'"
print("\nHighest SVM accuracy: " + str(highestAccuracy) + ", Parameters: C=" + str(c_val) + ", degree=" + str(d_val) + ", kernel=" + str(k_val) + ", decision_function_shape=" + str(w_val))
