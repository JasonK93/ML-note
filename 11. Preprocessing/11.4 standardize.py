from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler

def test_MinMaxScaler():
    '''
    test the method of MinMax Scaler
    :return: None
    '''
    X=[   [1,5,1,2,10],
      [2,6,3,2,7],
      [3,7,5,6,4,],
      [4,8,7,8,1] ]
    print("before transform:",X)
    scaler=MinMaxScaler(feature_range=(0,2))
    scaler.fit(X)
    print("min_ is :",scaler.min_)
    print("scale_ is :",scaler.scale_)
    print("data_max_ is :",scaler.data_max_)
    print("data_min_ is :",scaler.data_min_)
    print("data_range_ is :",scaler.data_range_)
    print("after transform:",scaler.transform(X))
def test_MaxAbsScaler():
    '''
    test the method of MaxAbs Scaler

    :return: None
    '''
    X=[   [1,5,1,2,10],
      [2,6,3,2,7],
      [3,7,5,6,4,],
      [4,8,7,8,1] ]
    print("before transform:",X)
    scaler=MaxAbsScaler()
    scaler.fit(X)
    print("scale_ is :",scaler.scale_)
    print("max_abs_ is :",scaler.max_abs_)
    print("after transform:",scaler.transform(X))
def test_StandardScaler():
    '''
    test the method of Standard Scaler
    :return: None
    '''
    X=[   [1,5,1,2,10],
      [2,6,3,2,7],
      [3,7,5,6,4,],
      [4,8,7,8,1] ]
    print("before transform:",X)
    scaler=StandardScaler()
    scaler.fit(X)
    print("scale_ is :",scaler.scale_)
    print("mean_ is :",scaler.mean_)
    print("var_ is :",scaler.var_)
    print("after transform:",scaler.transform(X))

if __name__=='__main__':
    test_MinMaxScaler()
    test_MaxAbsScaler()
    test_MaxAbsScaler()