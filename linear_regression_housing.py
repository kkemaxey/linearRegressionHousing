import torch
import numpy as nump
import pandas as pand
import matplotlib.pyplot as plt

#this is our data. we'll use pandas to organize it
df = pand.read_csv("https://raw.githubusercontent.com/selva86/datasets/refs/heads/master/BostonHousing.csv")
X = df['rm'].to_numpy()
Y = df['medv'].to_numpy()

# this is based on the normal equation of (X^t * X)^-(1) * (X^t * y)
Xcon = nump.concatenate((nump.ones((X.shape[0], 1)), X.reshape(-1,1)), axis = 1)
a = nump.matmul(nump.transpose(Xcon), Xcon)
b = nump.matmul(nump.transpose(Xcon), Y)
theta = nump.matmul(nump.linalg.inv(a), b)


#this is used to calculate the line using the probability
#the slope = covariance(x,y)/var(x)
#the y-intercept = y.mean - slope*x.mean
Xbar = X.mean()
Ybar = Y.mean()

cov = nump.cov(X,Y)
theta_1 = cov[0,1]/(X.var())
theta_0 = Ybar - theta_1*Xbar

#this is using machine learning to iterate through
w = 0.0
b = 0.0
learning_rate = 0.024
epochs = 25000

split_ratio = 0.8
split_index = int(len(X) * split_ratio)
indices = nump.random.permutation(len(X))

x_shuffled = X[indices]
y_shuffled = Y[indices]
x_train, x_test = x_shuffled[:split_index], x_shuffled[split_index:]
y_train, y_test = y_shuffled[:split_index], y_shuffled[split_index:]

n = float(len(x_train))

for epoch in range(epochs):
    y_pred = w*x_train + b
    cost = (1/(2*n))*sum((y_train-y_pred)**2)
    dw = -(2/n)*sum(x_train*(y_train-y_pred))
    db = -(2/n)*sum(y_train-y_pred)
    w = w - learning_rate*dw
    b = b - learning_rate*db
    
    print('Epoch {}, cost {}, m grad {}, b grad {}'.format(epoch, '%.3g' % cost,'%.3g' % dw, '%.3g' % db, '%.3g'))
 #   if epoch % 100 == 0:
  #    print(f'Epoch {epoch+1}/{epochs}, Cost: {cost}, w: {w}, b: {b}')

print(f'Final Cost: {cost}, w: {w}, b: {b}')
#print(f'Predicted Y values (on training data): {y_pred}')
#print(f'Actual Y values (on training data): {y_train}')
print(f'Final Weights: {w}')
print(f'Final Bias: {b}')

#Printing the results

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, alpha=0.7, label='Actual Data')
Y_pred = theta[0] + theta[1] * X

plt.plot(X, Y_pred, color='red', linewidth=2, label=f'Regression Line: Y = {theta[1]:.2f}X + {theta[0]:.2f}')
plt.title('Scatter Plot with Regression Line (Normal Formula Method)')
plt.xlabel('Number of Rooms (RM)')
plt.ylabel('Median Home Value (MEDV)')
plt.grid(True)
plt.legend()
plt.savefig('linearreg_normal.png')

xline = nump.linspace(X.min(), X.max(), 100)
yline = theta_0 + theta_1*xline

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, alpha=0.7, label='Actual Data')
plt.plot(xline, yline, color='black', linewidth=2, label=f'Regression Line: Y = {theta_1:.2f}X + {theta_0:.2f}')
plt.title('Scatter Plot with Regression Line (Probability and Statistics Method)')
plt.xlabel('Number of Rooms (RM)')
plt.ylabel('Median Home Value (MEDV)')
plt.grid(True)
plt.legend()
plt.savefig('lineaerreg_prob.png')

y_pred_ml = w * X + b

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, alpha=0.7, label='Actual Data')
plt.plot(X, y_pred_ml, color='blue', linewidth=2, label=f'Regression Line: Y = {w:.2f}X + {b:.2f}')
plt.title('Scatter Plot with Regression Line (Machine Learning Method)')
plt.xlabel('Number of Rooms (RM)')
plt.ylabel('Median Home Value (MEDV)')
plt.grid(True)
plt.legend()
plt.savefig('linearreg_ml.png')


#Saving the model
#nump.savetxt('model_weight.txt', w)
#nump.savetxt('model_bias', b)

nump.savez('linear_regression4.npz', weight=w, bias=b)
print("Model saved Successfully")
