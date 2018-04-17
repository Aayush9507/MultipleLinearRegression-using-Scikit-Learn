import numpy
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

iris = pd.read_csv('93cars.csv')
X = numpy.array(iris[['RPM','Horsepower']])
Y = numpy.array(iris['EngineSize'])

LinReg = LinearRegression()
LinReg.fit(X, Y)                # Training
pre = LinReg.predict(X)         # Predicting
er = numpy.sum((pre - Y) ** 2)  #Sum of Squared Errors
SquaredError = numpy.sqrt(er/len(pre))

print "R2", LinReg.score(X, Y)   # Accuracy
print "Sum of squared errors = ",er
print "y = ", LinReg.coef_, "x + ", LinReg.intercept_
print "Predicted", pre[:10]
plt.scatter(pre,Y, s=30, c='r' , marker='+', zorder=10)
plt.xlabel('Weight')
plt.ylabel('EngineSize')
plt.title('Prediction of Weight acc to Engine Size')
plt.plot(X, pre)
plt.show()
