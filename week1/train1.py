import  math
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



def cost_function(x,y,w,b):
    cost = 0
    m = len(x)
    for i in range(m):
        f_wb = ((w*x[i] + b) - y[i]) ** 2
        cost += f_wb
    J_wb = (1/2*m) * cost
    return J_wb

def gradient(x,y,w,b):
    dj_dw = 0
    dj_db = 0
    m = len(x)

    for i in range(m):
        dj_dw_xi = ((w*x[i] + b) - y[i]) * x[i]
        dj_db_xi = ((w*x[i] + b) - y[i])
        dj_dw += dj_dw_xi
        dj_db += dj_db_xi
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(x,y,w,b,alpha,iterations,cost_function,gradient):
    
    J_wb_history = []
    P_wb_history = []

    for i in range(iterations):
        dj_dw, dj_db = gradient(x,y,w,b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        if i < 10000:
            J_wb_history.append(cost_function(x,y,w,b))
            P_wb_history.append([w,b])
        if i % math.ceil(iterations/10) == 0:
            print(f'Iterations: {i:4}, Cost: {J_wb_history[-1]:0.2e}, dj_dw: {dj_dw:0.3e}, dj_db: {dj_db:0.3e}, w: {w:0.3e}, b: {b:0.3e}')
    return w,b,J_wb_history,P_wb_history

w_init = 0
b_init = 0
alpha = 0.01
iterations = 10000

w_final, b_final, J_hist, P_hist = gradient_descent(datax,datay,w_init,b_init,alpha,iterations,cost_function,gradient)

print(f'w,b are: {[w_final, b_final]} ')




