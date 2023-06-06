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



w = 0.391540
b = 0.411091

def compute_model_output(x, w, b):

    m = x.shape[0]
    f_wb = np.zeros(m) #np.zero(n) will return a one-dimensional numpy array with  𝑛 entries
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

tmp_f_wb = compute_model_output(datax, w, b,)

# Plot our model prediction
plt.plot(datax, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(datax, datay, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
