from sklearn.preprocessing import OneHotEncoder
def test_OneHotEncoder():
    '''
    test the method
    :return: None
    '''
    X=[   [1,2,3,4,5],
          [5,4,3,2,1],
          [3,3,3,3,3,],
          [1,1,1,1,1] ]
    print("before transform:",X)
    encoder=OneHotEncoder(sparse=False)
    encoder.fit(X)
    print("active_features_:",encoder.active_features_)
    print("feature_indices_:",encoder.feature_indices_)
    print("n_values_:",encoder.n_values_)
    print("after transform:",encoder.transform( [[1,2,3,4,5]]))
if __name__=='__main__':
    test_OneHotEncoder()