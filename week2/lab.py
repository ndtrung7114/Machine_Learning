import numpy as np
import matplotlib.pyplot as plt
import math
import copy

open_x = open('datax')
open_y = open('datay')

read_line_x = open_x.readlines()
read_line_y = open_y.readlines()
list_x = []
for i in range(len(read_line_x)):
    split_xi = read_line_x[i].split()
    for j in range(len(split_xi)):
        split_xi[j] = float(split_xi[j])
    list_x.append(split_xi)
datax = np.array(list_x)

def zscore_normalize_features(x):
    mu = np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    norm_x = (x - mu)/sigma

    return norm_x

for i in range(len(read_line_y)):
    read_line_y[i] = float(read_line_y[i])

datay = np.array(read_line_y)

datax = zscore_normalize_features(datax)


def compute_cost(x,y,w,b):
    cost = 0
    m =x.shape[0]
    for i in range(m):
        f_wb = ((np.dot(x[i],w) + b) - y[i] ) ** 2
        cost += f_wb
    cost /= (2*m)
    return cost

def comppute_gradient(x,y,w,b):
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
   
    for i in range(m):
        err = ((np.dot(x[i],w) + b) - y[i])
        for j in range(n):
            dj_dw[j] += err * x[i,j]
        dj_db += err
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(x,y,w,b,alpha,iteration,cost_function,gradient_function):
    j_histrory = []

    for i in range(iteration):
        dj_dw, dj_db =gradient_function(x,y,w,b)

        w -= alpha * dj_dw
        b -= alpha * dj_db
        
        if i < 10000:
            j_histrory.append(cost_function(x,y,w,b))

       
        if i % math.ceil(iteration/10) == 0:
            print(f'Iteration: {i:4d}, Cost: {j_histrory[-1]:8.8f}, w0: {w[0]:0.2f}, w1: {w[1]:0.2f}, w2: {w[2]:0.2f}, w3: {w[3]:0.2f}, w4: {w[4]:0.2f}, b: {b:0.2f}')
    return w,b,j_histrory

w = [0, 0, 0, 0, 0]
b = 0
alpha = 0.7
iterations = 1000

w_final, b_final, J_hist = gradient_descent(datax, datay, w, b, alpha, iterations, compute_cost, comppute_gradient)

print(f'w and b are {w_final}, {b_final:0.8f}')












