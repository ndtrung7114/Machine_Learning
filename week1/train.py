import math
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([5.1, 4.9, 4.7, 4.6, 5, 5.4])
y_train = np.array([3.5, 3, 3.2, 3.1, 3.6, 3.9])

def cost_function(x,y,w,b):
    cost = 0
    m = len(x_train)
    for i in range(m):
        f_wb = ((w*x[i] + b) - y[i]) ** 2
        cost += f_wb
    J_wb = (1/2*m) * cost
    return J_wb

def gradient(x,y,w,b):
    dj_db = 0
    dj_dw = 0
    m = len(x_train)
    for i in range(m):
        dj_db += ((w*x[i] + b) - y[i])
        dj_dw += ((w*x[i] + b) - y[i]) * x[i]
    dj_db /= m
    dj_dw /= m
    return dj_dw, dj_db
def gradient_descent(x,y,w_int,b_int,alpha,num_iters,cost_function,gradient):
    J_wb_his = []
    p_wb = []
   
    w = w_int
    b = b_int
    for i in range(num_iters):
        dj_dw, dj_db = gradient(x,y,w,b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        if i < 10000:
            J_wb_his.append(cost_function(x,y,w,b))
            p_wb.append([w,b])
        if i % math.ceil(num_iters/10) == 0:
            print(f'Iterations: {i:4}, cost: {J_wb_his[-1]:0.2e}, dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}, w: {w:0.3e}, b: {b:0.3e}')
    return w,b,J_wb_his,p_wb
w_int = 0
b_int = 0
alpha = 0.01
number_iterations = 10000
w_final, b_final, J_hist, p_hist = gradient_descent(x_train,y_train,w_int,b_int,alpha,number_iterations,cost_function,gradient)

print(f'Answer is w: {w_final}, b: {b_final}')

        