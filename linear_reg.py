
import numpy as np 
import scipy.stats as scp 

# ** Simple linear regression **

def prediction(alpha, beta, x_i):
    return beta * x_i + alpha 

# need to calculate the error of the alpha + beta preduiction 
# we dont watn to just add the total error as the prdiction 
#for x1 is too high and x2 is too low they cancel out
# so we added squared error 

def least_squares_error(x,y):
    beta = scp.pearsonr(x ,y) * np.std(y)/np.std(x)
    alpha = np.mean(y)- beta * np.mean(x)
    return alpha, beta 

def error(alpha, beta, x_i, y_i):
    "the error from predicted beta * x_i + alpha when the actual value is y_i"
    return y_i - prediction(alpha, beta, x_i)

    

 # R-SQUARED
# def total_sum_of_squares(y):
#     "total sum of squared variation of y_i from mean"
#     return sum(v ** 2 for v in de_mean(y))
 
#def r_squared(alpha, beta, x, y):
#    return 1.0 - (sum (sum_sq )
                  
# ** Gradient Descent **

def squared_error(x_i, y_i, theta):
    alpha, beta = theta 
    return error(alpha, beta, x_i, y_i) ** 2

def squared_error_gradient(x_i, y_i, theta):
    alpha, beta = theta
    return [-2 * error(alpha, beta, x_i, y_i), # alpha parial derivative
           -2 * error(error, beta, x_i, y_i) * x_i] # beta partial derivative

# Choose random value to start 

#lenas version  
    from scipy.stats.stats import pearsonr as pr
    def least_squares_error_l(x,y):
        Pearson = pr(x,y)
        r = Pearson[0]
        beta = r * np.std(y)/np.std(x)
        alpha = np.mean(y) - beta*np.mean(x)
        return alpha,beta
    
    coeff_l = least_squares_error_l(data_result.search_c_nc , data_result.direct)  
    print(coeff_l)


#random.seed(0)

#theta = [random.random(), random,random()]
#alpha, beta = minimize_stachastic(squared_error,
#                                  squared_error_gradient,
#                                  x, 
#                                  y, 
#                                  theta, 
#                                  0.0001]

