############################
## Name: Divyesh Rathod
## ASU ID: 1225916954
## Project 1 - Problem 2
############################


import numpy as np  
import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import *  
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score  
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import *
from sklearn.neighbors import KNeighborsClassifier 

# Setting the size of the models array
size = 6

# Creating arrays for storing test and combined accuracy for each model
test_accuracy_percent = np.zeros(size)
combined_accuracy_percent = np.zeros(size)


# Reading the dataset
df = pd.read_csv('heart1.csv')
df_numpy = df.to_numpy()

# Separating the features and the target variable
X = df_numpy[:, 0:13]
y = df_numpy[:, [13]].ravel()

# Split the dataset into training and testing sets using train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=18)


def train_data(classifier_type, normalize = False):             # Defining function for training and evaluating the model

    if(normalize == True):                                      # Normalize the features if normalize parameter is set to True
        sc = StandardScaler()  
        sc.fit(X_train)  
        X_train_sc = sc.transform(X_train)  
        X_test_sc = sc.transform(X_test)  
    else:
        X_train_sc = X_train
        X_test_sc = X_test

    
    # Fitting the classifier on the training data
    classifier_type.fit(X_train_sc, y_train) 
    
    # Combining the training and testing data
    X_combined = np.vstack((X_train_sc, X_test_sc))
    y_combined = np.hstack((y_train, y_test))
    
    # Predicting the target variable for the testing data
    y_pred = classifier_type.predict(X_test_sc)
    
    # Calculating the test accuracy
    test_acc = accuracy_score(y_test, y_pred)

    # Predicting the target variable for the combined data
    y_combined_pred = classifier_type.predict(X_combined)
    
    # Calculating the combined accuracy
    combined_acc = accuracy_score(y_combined, y_combined_pred)
    
    # Rounding the accuracies to 2 decimal places
    test_acc_per = test_acc*100
    test_acc_per = round(test_acc_per,2)
    combined_acc_per = combined_acc*100
    combined_acc_per = round(combined_acc_per,2)

    # Returning the test and combined accuracy percentages
    return test_acc_per, combined_acc_per


#Perceptron Classifier
classifier = Perceptron(max_iter=15, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=18, verbose=False)
test_accuracy_percent[0], combined_accuracy_percent[0] = train_data(classifier, True)

#Logistic regression classifier
classifier = LogisticRegression(C = 10, solver='liblinear', multi_class='ovr', random_state = 18)
test_accuracy_percent[1],  combined_accuracy_percent[1] = train_data(classifier, True)

#Support vector classifier
classifier = SVC(kernel = 'linear', C=1, random_state = 18)
test_accuracy_percent[2], combined_accuracy_percent[2] = train_data(classifier, True)

#Decision tree classifier
classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, random_state = 18)
test_accuracy_percent[3], combined_accuracy_percent[3] = train_data(classifier, True)

#Random Forest classifier
classifier = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=18, n_jobs=1)
test_accuracy_percent[4], combined_accuracy_percent[4] = train_data(classifier, True)

#K Nerest neighbors classifier
classifier = KNeighborsClassifier(n_neighbors=1,p=1,metric='minkowski')
test_accuracy_percent[5], combined_accuracy_percent[5] = train_data(classifier)

print("Table of Prediction Percentages\n")


# Create a pandas dataframe 
percentage_matrix = list(zip(test_accuracy_percent,combined_accuracy_percent))
df = pandas.DataFrame(percentage_matrix, index = ['Perceptron', 'Logistic Regression', 'Support Vector', 'Decision Tree', 'Random Forest', 'K Nearest Neighbor'], columns = [ 'Test Accuracy in %','Combined Accuracy in %'])

#Display the accuracy table
print(df)