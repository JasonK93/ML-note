from sklearn.metrics import mean_absolute_error,mean_squared_error

def test_mean_absolute_error():

    y_true=[1,1,1,1,1,2,2,2,0,0]
    y_pred=[0,0,0,1,1,1,0,0,0,0]

    print("Mean Absolute Error:",mean_absolute_error(y_true,y_pred))
def test_mean_squared_error():

    y_true=[1,1,1,1,1,2,2,2,0,0]
    y_pred=[0,0,0,1,1,1,0,0,0,0]

    print("Mean Absolute Error:",mean_absolute_error(y_true,y_pred))
    print("Mean Square Error:",mean_squared_error(y_true,y_pred))

if __name__=="__main__":
    test_mean_absolute_error()
    test_mean_squared_error() 