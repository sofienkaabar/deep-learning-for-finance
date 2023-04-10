# Defining a variable
x = 10
y = 5

# Writing a constant
6

# Declaring my_variable
my_variable = 1

# Declaring My_variable
My_variable = 2

# The variable my_variable is different from My_variable

# Returns a SyntaxError
1x = 5

# Valid declaration
x1 = 5

# Valid declaration
x1x = 5

# Returns a SyntaxError
x-y = 5

# Valid declaration
x_y = 5

# Recommended name
ma_lookback = 10

# Not recommended name
the_lookback_on_that_moving_average = 10

# Creating a variable that holds an integer
my_integer = 1

# Creating a variable that holds a float number
my_float_number = 1.2

# Using the built-in Python function type() to verify the variables
type(my_integer)
type(my_float_number)

# Outputting the phrase "Hello World"
print('Hello World')

# Make a statement that the type of my_integer is integer
type(my_integer) is int

# Make a statement that the type of my_float_number is float
type(my_float_number) is float

# Make a statement that the type of my_integer is float
type(my_integer) is float

'''
Intuitively, the two first statements will return True as they are 
indeed true. The third statement is False as the variable my_integer
is an integer and not a float number

'''

# Arithmetic operator - Addition
1 + 1 # The line outputs 2

# Arithmetic operator - Subtraction
1 - 1 # The line outputs 0

# Arithmetic operator - Multiplication
2 * 2 # The line outputs 4

# Arithmetic operator - Division
4 / 2 # The line outputs 2.0 as a float number

# Arithmetic operator - Exponents
2 ** 4 # The line outputs 16

# Comparison operator - Equality
2 == 2 # The line outputs True

# Comparison operator - Non equality
2 != 3 # The line outputs True

# Comparison operator - Greater than
2 > 3 # The line outputs False

# Comparison operator - Greater than or equal to
2 >= 2 # The line outputs True

# Comparison operator - Less than
2 < 3 # The line outputs True

# Comparison operator - Less than or equal to
2 <= 2 # The line outputs True


# Logical operator - and
2 and 1 < 4 # The line outputs True
2 and 5 < 4 # The line outputs False

# Logical operator - or
2 or 5 < 4 # The line outputs 2 which is the integer less than 4

# Declaring two variables x and y and assigning them values
x = 10
y = 2.5

# Checking the types of the variables
type(x) # Returns int
type(y) # Returns float

# Taking x to the power of y and storing it in a variable z
z = x ** y # Returns 316.22

# Checking if the result is greater than or equal to 100
z >= 100 # Returns True as 316.22 >= 100

# Declaring the variables
a = 9
b = 2

# First condition (specific)
if a > b:
    
    print('a is greater than b')

# Second condition (specific)    
elif a < b:
    
    print('a is lower than b')

# Third condition (general)    
else:
    
    print('a is equal to b')


# Using a for loop
for i in range(1, 5):
    
    print(i)
    
# Using a while loop  
i = 1    
while i < 5:

    print(i)
    i = i + 1

# Creating the time series
time_series = [1, 3, 5, 2, 4, 1, 6, 4, 2, 4, 4, 4]

for i in range(len(time_series)):
    
    # The condition where the current price rose
    if time_series[i] > time_series[i - 1]:
        
        print(1)
    
    # The condition where the current price fell
    elif time_series[i] < time_series[i - 1]:
        
        print(-1) 
        
    # The condition where the current price hasn't changed
    else:
        
        print(0)


# The import statement must be followed by the name of the library
import numpy

# Optionally, you can give the library a shortcut for easier references
import numpy as np


# Importing one function from a library
from math import sqrt


# Defining the function sum_operation and giving it two arguments
def sum_operation(first_variable, second_variable):
    
    # Outputing the sum of the two variables
    print(first_variable + second_variable)

# Calling the function with 1 and 3 as arguments
sum_operation(1, 3) # The output of this line is 4


# Importing the library    
import math

# Using the natural logarithm function
math.log(10)

# Using the exponential function (e)
math.exp(3)

# Using the factorial function
math.factorial(50)


# Defining a function to sum two variables and return the result
def sum_operation(first_variable, second_variable):
    
    # The summing operation is stored in a variable called final_sum
    final_sum = first_variable + second_variable
    
    # The result is returned
    return final_sum
    
# Create a new variable that holds the result of the function    
summed_value = sum_operation(1, 2)

# Use the new variable in a new mathematical operation and store the result
double_summed_value = summed_value * 2


# Defining a function to square the result gotten from the sum_operation function
def square_summed_value(first_variable, second_variable):
    
    # Calling the nested sum_operation function and storing its result
    final_sum = sum_operation(first_variable, second_variable)
    
    # Creating a variable that stores the square of final_sum
    squared_sum = final_sum ** 2

    # The result is returned    
    return squared_sum

# Create a new variable that holds the result of the function   
squared_summed_value = square_summed_value(1, 2)

# Will not output a SyntaxError if executed
my_range = range(1, 10)

# Will output a SyntaxError is executed
my_range = range(1, 10
                 
# Importing the required library to create an array
import numpy as np

# Creating a two-column list with 8 rows
my_time_series = [(1, 3), 
                  (1, 4), 
                  (1, 4), 
                  (1, 6), 
                  (1, 4), 
                  (0, 2), 
                  (1, 1), 
                  (0, 6)]

# Transforming the list into an array
my_time_series = np.array(my_time_series)               
                 
# Defining the function
def division(first_column, second_column):
    
    # Looping through the length of the created array
    for i in range(len(my_time_series)):
        
        # Division operation and storing it in the variable x
        x = my_time_series[i, first_column] / my_time_series[i + 1, second_column]
        
        # Outputting the result
        print(x)

# Calling the function
division(0, 1)                 
                 
# Defining the function
def division(first_column, second_column):
    
    # Looping through the length of the created array    
    for i in range(len(my_time_series)):
        
        # First part of the exception handling
        try:

            # Division operation and storing it in the variable x
            x = my_time_series[i, first_column] / my_time_series[i + 1, second_column]
            
            # Outputting the result            
            print(x)
        
        # Exception handling of a specific error     
        except IndexError:

            # Ignoring (passing) the error
            pass

# Calling the function
division(0, 1)                 
              

# Creating a data frame
my_data_frame = pd.DataFrame({'first_column' : [1, 2, 3], 
                              'second_column' : [4, 5, 6]})

# Creating an array
my_array = np.array([[1, 4], [2, 5], [3, 6]])    

# To transform my_data_frame into my_new_array
my_new_array = np.array(my_data_frame)

# To transform my_array into my_new_data_frame
my_new_data_frame = pd.DataFrame(my_array)

first_array  = np.array([ 1,  2,  3,  5,   8,  13])
second_array = np.array([21, 34, 55, 89, 144, 233])

# Reshaping the arrays so they become compatible in multidimensional manipulation
first_array  = np.reshape(first_array, (-1, 1))
second_array = np.reshape(second_array, (-1, 1))

# Concatenating both arrays by columns
combined_array = np.concatenate((first_array, second_array), axis = 1)

# Concatenating both arrays by rows
combined_array = np.concatenate((first_array, second_array), axis = 0)

first_data_frame  = pd.DataFrame({'first_column'  : [ 1,  2,  3], 
                                  'second_column' : [ 4,  5,  6]})
second_data_frame = pd.DataFrame({'first_column'  : [ 7,  8,  9], 
                                  'second_column' : [10, 11, 12]})

# Concatenating both data frames by columns
combined_data_frame = pd.concat([first_data_frame, second_data_frame], axis = 1)

# Concatenating both data frames by rows
combined_data_frame = pd.concat([first_data_frame, second_data_frame], axis = 0)


# Defining a one-dimensional array
my_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Referring to the first value of the array
my_array[0] # Outputs 1

# Referring to the last value of the array
my_array[-1] # Outputs 1​0

# Referring to the fifth value of the array
my_array[6] # Outputs 7

# Referring to the first three values of the array
my_array[0:3] # Outputs array([1, 2, 3])
my_array[:3]  # Outputs array([1, 2, 3])

# Referring to the last three values of the array
my_array[-3:] # Outputs array([8, 9, 10])

# Referring to all the values as of the second value
my_array[1:] # Outputs array([2, 3, 4, 5, 6, 7, 8, 9, 10])

# Defining a multi-dimensional array
my_array = np.array([[ 1,  2,  3,  4,  5], 
                     [ 6,  7,  8,  9, 10], 
                     [11, 12, 13, 14, 15]])

# Referring to the first value and second column of the array
my_array[0, 1] # Outputs 2

# Referring to the last value and last column of the array
my_array[-1, -1] # Outputs 15

# Referring to the third value and second to last column of the array
my_array[2, -2] # Outputs 14

# Referring to the first three and fourth column values of the array
my_array[:, 2:4] # Outputs array([[3, 4], [8, 9], [13, 14]])

# Referring to the last two values and fifth column of the array
my_array[-2:, 4] # Outputs array([10, 15])

# Referring to all the values and all the columns up until the second row
my_array[:2, ] # Outputs array([[ 1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

# Referring to the last row with all the columns
my_array[-1:, :] # Outputs array([[11, 12, 13, 14, 15]])



# Defining a one-dimensional data frame
my_df= pd.DataFrame({'first_column': [1, 2, 3, 4, 5, 
                                      6, 7, 8, 9, 10]})

# Referring to the first value of the data frame
my_df.iloc[0]['first_column'] # Outputs 1

# Referring to the last value of the data frame
my_df.iloc[-1]['first_column'] # Outputs 10

# Referring to the fifth value of the data frame
my_df.iloc[6]['first_column'] # Outputs 7

# Referring to the first three values of the data frame
my_df.iloc[0:3]['first_column'] # Outputs ([1, 2, 3])

# Referring to the last three values of the data frame
my_df.iloc[-3:]['first_column'] # Outputs ([8, 9, 10])

# Referring to all the values as of the second value
my_df.iloc[1:]['first_column'] # Outputs ([2, 3, 4, 5, 6, 7, 8, 9, 10])

# Defining a multi-dimensional data frame
my_df  = pd.DataFrame({'first_column'  : [ 1,  6,  11], 
                       'second_column' : [ 2,  7,  12],
                       'third_column'  : [ 3,  8,  13],                       
                       'fourth_column' : [ 4,  9,  14],                       
                       'fifth_column'  : [ 5,  10, 15]})

# Referring to the first value and second column of the data frame
my_df.iloc[0]['second_column'] # Outputs 2

# Referring to the last value and last column of the data frame
my_df.iloc[-1]['fifth_column'] # Outputs 15

# Referring to the third value and second to last column of the data frame
my_df.iloc[2]['fourth_column'] # Outputs 14

# Referring to the first three and fourth column values of the data frame
my_df.iloc[:][['third_column', 'fourth_column']]

# Referring to the last two values and fifth column of the data frame
my_df.iloc[-2:]['fifth_column'] # Outputs ([10, 15])

# Referring to all the values and all the columns up until the second row
my_df.iloc[:2,] # Outputs ([[ 1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

# Referring to the last row with all the columns
my_df.iloc[-1:,]  # Outputs ([[11, 12, 13, 14, 15]])

import datetime
import pytz
import pandas                    as pd
import MetaTrader5               as mt5
import matplotlib.pyplot         as plt
import numpy                     as np

frame_H1   = mt5.TIMEFRAME_H1
frame_D1   = mt5.TIMEFRAME_D1
frame_W1   = mt5.TIMEFRAME_W1

now = datetime.datetime.now()

assets = ['EURUSD', 'USDCHF', 'GBPUSD', 'USDCAD', 'AUDUSD', 'NZDUSD', 'EURGBP', 'EURCHF', 'EURCAD', 'EURAUD']
     
def mass_import(asset, time_frame):
                
    if time_frame == 'H1':
        data = get_quotes(frame_H1, 2012, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)        
        
    if time_frame == 'D1':
        data = get_quotes(frame_D1, 2000, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)        
 
    if time_frame == 'W1':
        data = get_quotes(frame_W1, 2000, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)        
                   
    return data 

def get_quotes(time_frame, year = 2005, month = 1, day = 1, asset = "EURUSD"):
        
    if not mt5.initialize():
        
        print("initialize() failed, error code =", mt5.last_error())
        
        quit()
    
    timezone = pytz.timezone("Europe/Paris")
    
    time_from = datetime.datetime(year, month, day, tzinfo = timezone)
    
    time_to = datetime.datetime.now(timezone) + datetime.timedelta(days=1)
    
    rates = mt5.copy_rates_range(asset, time_frame, time_from, time_to)
    
    rates_frame = pd.DataFrame(rates)

    return rates_frame  

# Calling the mass_import function and storing it into a variable
eurusd_data = mass_import(0, 'H1')

# Importing the excel file into the Python interpreter
my_data = pd.read_excel('eurusd_data.xlsx')











                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 









