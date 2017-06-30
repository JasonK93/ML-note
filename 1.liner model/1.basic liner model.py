# import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, discriminant_analysis, cross_validation


'''
load_data : get the diabetes data from the pkg of sklearn
return:
        1 array for the regression problem.
        train_data, test_data, train_value, test_value
'''

def load_data():

    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data,diabetes.target,
		test_size=0.25,random_state=0)


'''
test_LR: the code that train the model
param data: *data is a parameter that can change
Return: None
'''
def test_LinearRegression(*data):

    X_train,X_test,y_train,y_test=data
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print('Coefficients:{0}, intercept {1}'.format(regr.coef_,regr.intercept_))
    print("Residual sum of squares: {0}".format(np.mean((regr.predict(X_test) - y_test) ** 2)))
    print('Score: {0}'.format(regr.score(X_test, y_test)))
# the main function
if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data()
    test_LinearRegression(X_train,X_test,y_train,y_test)