#The optimal values of m and b can be actually calculated with way less effort than doing a linear regression. 
#this is just to demonstrate gradient descent

from numpy import *
import pandas as pd
# y = mx + b
# m is slope, b is y-intercept

# function to compute error for a line, the standard function is  https://spin.atomicobject.com/wp-content/uploads/linear_regression_error1.png

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

#this is the function for gradient descent.
def step_gradient(b_current, m_current, points, learningRate):
    #lets start by initializing gradient for m and b to be zero, which means y_hat doesn't change based on x_axis, However this will not be the case as we will update these gradient based on our error function.
    b_gradient = 0
    m_gradient = 0
	#N is the total number of observation available to us.
    N = float(len(points))
	# now for each observation x in points, and label y in points, we will calculate the gradient which will den be utilized for updating current using partial derivatives https://spin.atomicobject.com/wp-content/uploads/linear_regression_gradient1.png
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
	#after for loop we recieved update m and b gradient and using updated gradients we then update our current m and b for y_hat = m*x + b
	#learningRate determine how fast we need to reduce our current m and b.
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

# function to run step gradient method, it executes step_gradient function for the number of iterations defined
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

#main function
def run():
    points = pd.read_csv("data.csv", delimiter=",").values
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 5000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print("Running...")
	#Function to provide update optimized values for b and m
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

if __name__ == '__main__':
    run()
