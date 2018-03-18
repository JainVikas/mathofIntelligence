setwd("C:/Users/vikas/Desktop/Learning/mathofIntelligence/Implement_gradient_descent")

# y = mx + b
# m is slope, b is y-intercept

# function to compute error for a line, the standard function is  https://spin.atomicobject.com/wp-content/uploads/linear_regression_error1.png
compute_error_for_line_given_points <- function(b, m, points){
  totalError <- 0
  for (i in seq(1, nrow(points)))
    {
    x = points[i, 1]
    y = points[i, 2]
    totalError <- totalError +  (y - (m * x + b)) ** 2
    }
 return(totalError/ nrow(points))
}

#this is the function for gradient descent.
step_gradient<- function(b_current, m_current, points, learningRate){
  #lets start by initializing gradient for m and b to be zero, which means y_hat doesn't change based on x_axis, However this will not be the case as we will update these gradient based on our error function.
  b_gradient <- 0
  m_gradient <- 0
  #N is the total number of observation available to us.
  N <- nrow(points)
# now for each observation x in points, and label y in points, we will calculate the gradient which will den be utilized for updating current using partial derivatives https://spin.atomicobject.com/wp-content/uploads/linear_regression_gradient1.png
  for (i in seq(1, nrow(points))){
    x <- points[i,1]
    y <- points[i, 2]
    b_gradient = b_gradient -(2/N) * (y - ((m_current * x) + b_current))
    m_gradient = m_gradient -(2/N) * x * (y - ((m_current * x) + b_current))
  }
  #after for loop we recieved update m and b gradient and using updated gradients we then update our current m and b for y_hat = m*x + b
  #learningRate determine how fast we need to reduce our current m and b.
  new_b = b_current - (learningRate * b_gradient)
  new_m = m_current - (learningRate * m_gradient)
return(list("new_b"=new_b, "new_m"= new_m))
  }


# function to run step gradient method, it executes step_gradient function for the number of iterations defined
gradient_descent_runner <- function(points, starting_b, starting_m, learning_rate, num_iterations){
  b = starting_b
  m = starting_m
  for (i in seq(num_iterations)){
    response <- step_gradient(b, m, points, learning_rate)
    b= response$new_b
    m= response$new_m
    }
  
return (list("b"=b, "m"=m))
}

points = read.csv("data.csv", header = FALSE)
learning_rate = 0.0001
initial_b = 0 # initial y-intercept guess
initial_m = 0 # initial slope guess
num_iterations = 5000
print(sprintf("After %s iterations b = %s, m = %s, error = %s",num_iterations, initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
print("Running...")
#Function to provide update optimized values for b and m
result = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
print(sprintf("After %s iterations b = %s, m = %s, error = %s",num_iterations, result$b, result$m, compute_error_for_line_given_points(result$b, result$m, points)))
