from sklearn.decomposition import DictionaryLearning

def test_DictionaryLearning():
    '''
    test the DL method
    :return: None
    '''
    X=[[1,2,3,4,5],
       [6,7,8,9,10],
       [10,9,8,7,6,],
       [5,4,3,2,1] ]
    print("before transform:",X)
    dct=DictionaryLearning(n_components=3)
    dct.fit(X)
    print("components is :",dct.components_)
    print("after transform:",dct.transform(X))

if __name__=='__main__':
    test_DictionaryLearning()