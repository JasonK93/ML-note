from sklearn.preprocessing import Binarizer
def test_Binarizer():
    '''
    test Binatizer method
    :return: None
    '''
    X=[   [1,2,3,4,5],
          [5,4,3,2,1],
          [3,3,3,3,3,],
          [1,1,1,1,1] ]
    print("before transform:",X)
    binarizer=Binarizer(threshold=2.5)
    print("after transform:",binarizer.transform(X))

if __name__=='__main__':
    test_Binarizer()