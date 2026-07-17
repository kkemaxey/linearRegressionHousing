import numpy as nump

model = nump.load('linear_regression.npz')
data = nump.load('test_data.npz')
w = model['weight']
b = model['bias']

y_pred = w * data['x_test'] + b
mse = nump.mean((data['y_test'] - y_pred)**2)
rmse = nump.sqrt(mse)

SS_res = nump.sum((data['y_test'] - y_pred)**2)
SS_tot = nump.sum((data['y_test'] - data['y_test'].mean())**2)
r_squared = 1 - (SS_res/SS_tot)

rmse_real = rmse * 1000
r_squared_real = r_squared * 100
print(f'The model is off by ${rmse_real:.2f} of value on average')
print(f'The model accounts for {r_squared_real:.1f}% of the variation in home value')
