############################
## Name: Divyesh Rathod
## ASU ID: 1225916954
## Project 1 - Problem 1
############################

import numpy as np                                     
import pandas as pd                              
import seaborn as sns
import matplotlib.pyplot as plt

def correlation(dataframe):                               # Define function to calculate correlation, cross-covariance, and most highly correlated features
    column_names = dataframe.columns.tolist()             # Create a list of column names from the dataframe
    size = dataframe.values.shape[1]                      # Determine number of features in the dataframe
    
    correlation_mat = dataframe.corr().round(2)                    # Calculate correlation matrix and round to 2 decimal places
    covariance = dataframe.cov().round(2)                          # Calculate covariance matrix and round to 2 decimal places
    print('Initial Correlation Matrix')
    print(correlation_mat)                                         # Print initial correlation matrix
    print('\n')
    print('Covariance Matrix')
    print(covariance)                                              # Print covariance matrix
    print('\n')
    
    corr_matrix_test = dataframe.corr().abs()                                                      # Create a matrix of absolute correlations for feature selection
    upper = corr_matrix_test.where(np.triu(np.ones(corr_matrix_test.shape), k=1).astype(np.bool))  # Select only upper triangle of matrix
    highly_correlated = upper.stack().nlargest(10)                                                 # Select top 10 highly correlated features
    print('Most highly correlated with each other')
    print(highly_correlated)                                                                       # Print top highly correlated features
    print('\n')
    
    array_index = np.zeros((size))                         # Create an array for the index of the highly correlated feature
    array_cor = np.zeros((size))                           # Create an array for the correlation value between each feature and highly correlated feature
    array_cor_temp = np.zeros((size-1))                    # Create an array for correlation values between each feature and highly correlated feature for use in Table 4
    array_cor_index = np.zeros((size-1))

    # Remove the number 1 from the diagonal elements of the correlation matrix
    array = np.zeros((size, size))                     
    for i in range(size):
        for j in range(size):
            array[i][i] = 1
    diagonal_mat = correlation_mat - array                   # Subtract the identity matrix from correlation matrix to remove diagonal of ones

    # Find the high correlation for all the variables with every other variable and store it in an array
    for i in range(size):
        temp = diagonal_mat.iloc[[i]].to_numpy()              # Select row i of diagonal_mat as numpy array
        index_value = np.argmax(abs(temp))                    # Find index of maximum absolute correlation value
        array_index[i] = index_value                          # Store index of highly correlated feature
        array_cor[i] = round(temp[0][index_value],2)          # Store correlation value between feature i and highly correlated feature
        if(i!=size-1):
            array_cor_temp[i] = temp[0][size-1]         # Store correlation value between feature i and a1p2 for use in Table 4

    
# Create a table of highly correlated with a1p2 variable    
    table1 = list(zip( column_names, array_cor_temp))
    df4 = pd.DataFrame(table1, index=list(range(size-1)), columns=['Variable', 'Correlation with a1p2'])
    print('Correlation Matrix with a1p2')
    print(df4)
    print('\n')
    
# Create a table of variables that are highly correlated with each other
    column_names_2 = []
    for i in range(size):
        column_names_2.append(column_names[int(array_index[i])])

    table2 = list(zip(column_names, column_names_2, array_cor))
    df5 = pd.DataFrame(table2, index = list(range(size)), columns = ['Variable 1', 'Variable 2', 'Correlation'])
    print('Highest Correlation of each variable')
    print(df5)
    print('\n')

# Define a function to create pair plots of the data
def pairplot(df):                                       
    sns.pairplot(df,height=2.5,hue = 'a1p2', markers=['+', 'x'])
    plt.show()
    
### Main code

df = pd.read_csv('heart1.csv')
correlation(df)
pairplot(df)