from model.solvepolynomial import SGDRidgeRegression,SGDLinearRegression
import matplotlib.pyplot as plt
import numpy as np

import math

x = np.linspace(-1,1,20).reshape((-1,1))
y = np.sin(x)
model = SGDLinearRegression(learning_rate=0.5,batch_size=6,regularization_strength=2,n_epochs=100)
# model = SGDRidgeRegression(learning_rate=0.5,batch_size=6,regularization_strength=2,n_epochs=10000)
model.fit(x,y)

# model.predict(x[0:2])
plt.plot(x,y)
plt.scatter(x,model.predict(x))

plt.show()

