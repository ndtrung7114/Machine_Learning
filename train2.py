import math
import numpy as np
import matplotlib.pyplot as plt

f = open('datax')
t = open('datay')

datax = f.readlines()
datay = t.readlines()

for i in range(len(datax)):
    datax[i] = float(datax[i])
datax = np.array(datax)

for j in range(len(datay)):
    datay[j] = float(datay[j])
datay = np.array(datay)

def cost_funtion(x,y,w,b):
    m = len(datax)
    cost = 0
    for i in range(m):
        f_wb = ((w*x[i] + b) - y[i]) ** 2
        cost += f_wb
    J_wb = (1/2*m) * cost
    return J_wb

def gradient(x,y,w,b):
    dj_dw = 0
    dj_db = 0
    m = len(datax)
    for i in range(m):
        dj_dw_xi = ((w*x[i] + b) - y[i])* x[i]
        dj_db_xi = ((w*x[i] + b) - y[i])
        dj_dw += dj_dw_xi
        dj_db += dj_db_xi
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(x,y,w,b,alpha,iterations,cost_function,gradient):
    J_history = []
    P_history = []

    
    for i in range(iterations):
        dj_dw, dj_db = gradient(x,y,w,b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        if i < 10000:
            J_history.append(cost_function(x,y,w,b))
            P_history.append([w,b])
        if i % math.ceil(iterations/10) == 0:
            print(f'Iteration: {i:4}, Cost: {J_history[-1]:0.2e}, Dj_Dw: {dj_dw:0.3e}, Dj_Db: {dj_db:0.3e}, w: {w:0.3e}, b: {b:0.3e}')
    return w,b,J_history,P_history

w = 0
b = 0
alpha = 0.01
iteration = 10000
w_final, b_final, J_his, P_his = gradient_descent(datax,datay,w,b,alpha,iteration,cost_funtion,gradient)

print(f"Fianl w and b are : {w_final:4f},{b_final:4f}")
