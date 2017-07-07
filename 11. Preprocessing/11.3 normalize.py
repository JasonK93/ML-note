from sklearn.preprocessing import Normalizer
def test_Normalizer():
    '''
    test the method
    :return: None
    '''
    X=[   [1,2,3,4,5],
          [5,4,3,2,1],
          [1,3,5,2,4,],
          [2,4,1,3,5] ]
    print("before transform:",X)
    normalizer=Normalizer(norm='l2')
    print("after transform:",normalizer.transform(X))

if __name__=='__main__':
    test_Normalizer()