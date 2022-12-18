# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

'''
The relationship I have modeled was the amount of salary earned (per month)
as a mobile developer based on years of experience.
algorithm used for training set: y = 2(x)**2+x+3
x = years of experience
y = money earned (in rands) in thousands
'''

# Training set
# years of experience
x_train = [[1], [2], [3], [4], [5]]
# amount earned in rands in thousands
y_train = [[6], [13], [24], [58], [81]]

# Testing set
# years of experience
x_test = [[1], [2], [5], [7]]
# amount earned in rands in thousands
y_test = [[8], [18], [70], [95]]

# Setting evenly spaced numbers
xx = np.linspace(0, 70, 100)

# Setting the degree for the model
features = PolynomialFeatures(degree=2)

x_quadtrain = features.fit_transform(x_train)
x_quadtest = features.transform(x_test)

# Training and Testing
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(x_quadtrain, y_train)
xx_quadratic = features.transform(xx.reshape(xx.shape[0], 1))

# Initialing and plotting of graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='-')
plt.title('Monthly salary of mobile developers \n based on years of experience')
plt.xlabel('Years of experience')
plt.ylabel('Amount earned (rands) \n in thousands')
plt.axis([0, 8, 0, 150])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()