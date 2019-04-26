import pystan
import numpy as np

ocode = """
data {
    int<lower=1> N;
    real y[N];
}
parameters {
    real mu;
    real standard_devation;
}
model {
    y ~ normal(mu, standard_devation);
}
"""
sm = pystan.StanModel(model_code=ocode)

data = np.random.normal(size=20, scale=100)
print("THIS is data",data)
print(np.mean(data))

op = sm.optimizing(data=dict(y=data, N=len(data)))

print(op)