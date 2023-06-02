import math
import numpy as np
import matplotlib.pyplot as plt

f = open('datax')
t = open('datay')


datax = f.readlines()
datay = t.readlines()
for i in range(len(datax)):
    datax[i] = float(datax[i])
for k in range(len(datay)):
    datay[k] = float(datay[k])
datax = np.array(datax)
datay = np.array(datay)


def compute_cost_function(x,y,w,b):
    cost = 0
    m = len(x)
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i])**2
    J_wb = (1/2*m) * cost
    return J_wb

def compute_gradient(x,y,w,b):
    m = len(x)
    dj_dw = 0 #dao ham cua J theo w
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_xi = (f_wb - y[i]) * x[i]
        dj_dw += (dj_dw_xi)
        dj_db_xi = (f_wb - y[i])
        dj_db += dj_db_xi 
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    J_history = []
    p_history = []

    w = w_in
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x,y,w,b)
        b -= alpha * dj_db
        w -= alpha * dj_dw

        if i < 10000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        if i % math.ceil(num_iters/10) == 0:
            print(f'Iterations: {i:4}, Cost: {J_history[-1]:0.2e}, dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}, w: {w:0.3e}, b: {b: 0.3e}')

    return w,b,J_history,p_history

w_init = 0
b_init = 0
iterations = 10000
tmp_alpha = 0.01

w_final, b_final, J_hist, p_hist = gradient_descent(datax, datay, w_init, b_init, tmp_alpha, iterations, compute_cost_function, compute_gradient)

print((f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})"))          
                 