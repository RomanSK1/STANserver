from numpy import *
Nobs = 20
phi_true = random.uniform(0.0,2*3.14, size=Nobs)
center_x_true = 0.7
center_y_true = 0.8
radius_true = 2.0
eps_true = 0.01
x_true =  radius_true*cos(phi_true) + center_x_true 
y_true =  radius_true*sin(phi_true) + center_y_true 
x_obs = x_true + random.normal(0, eps_true, size=Nobs)
y_obs = y_true + random.normal(0, eps_true, size=Nobs)

from matplotlib import pyplot as plt
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.scatter(phi_true , x_obs, c=x_true, marker='o')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Z')
plt.show()

model = """
data {
   int<lower=4> N; // Number of data points
   real phi[N];      // the 1st predictor
   real x[N];      // the outcome
   real y[N];      // the outcome
}
parameters {
   real center_x;
   real center_y;     
   real radius;     
   real<lower=0> eps;       // dispersion
}
model {
   for (i in 1:N){
      x[i] ~ normal( radius*cos(phi[i]) + center_x , eps);
      y[i] ~ normal( radius*sin(phi[i]) + center_y , eps);
   }
}"""

data = {'N':Nobs, 'phi':phi_true, 'x':x_true, 'x_obs':x_obs, 'y':y_true, 'y_obs':y_obs}


import pystan
fit = pystan.stan(model_code=model, data=data, iter=1000, chains=4)

print(fit)


