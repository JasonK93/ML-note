from sklearn.feature_selection import  RFE,RFECV
from sklearn.svm import LinearSVC
from sklearn.datasets import  load_iris
from  sklearn import  cross_validation

def test_RFE():
    '''
    test the method of RFE, the number of feature aim to 2
    :return: None
    '''
    iris=load_iris()
    X=iris.data
    y=iris.target
    estimator=LinearSVC()
    selector=RFE(estimator=estimator,n_features_to_select=2)
    selector.fit(X,y)
    print("N_features %s"%selector.n_features_)
    print("Support is %s"%selector.support_)
    print("Ranking %s"%selector.ranking_)
def test_RFECV():
    '''
    test the method of RFECV
    :return:  None
    '''
    iris=load_iris()
    X=iris.data
    y=iris.target
    estimator=LinearSVC()
    selector=RFECV(estimator=estimator,cv=3)
    selector.fit(X,y)
    print("N_features %s"%selector.n_features_)
    print("Support is %s"%selector.support_)
    print("Ranking %s"%selector.ranking_)
    print("Grid Scores %s"%selector.grid_scores_)
def test_compare_with_no_feature_selection():
    '''
    compare the result before the selection and after
    :return: None
    '''
    iris=load_iris()
    X,y=iris.data,iris.target
    estimator=LinearSVC()
    selector=RFE(estimator=estimator,n_features_to_select=2)
    X_t=selector.fit_transform(X,y)
    X_train,X_test,y_train,y_test=cross_validation.train_test_split(X, y,
                test_size=0.25,random_state=0,stratify=y)
    X_train_t,X_test_t,y_train_t,y_test_t=cross_validation.train_test_split(X_t, y,
                test_size=0.25,random_state=0,stratify=y)
    clf=LinearSVC()
    clf_t=LinearSVC()
    clf.fit(X_train,y_train)
    clf_t.fit(X_train_t,y_train_t)
    print("Original DataSet: test score=%s"%(clf.score(X_test,y_test)))
    print("Selected DataSet: test score=%s"%(clf_t.score(X_test_t,y_test_t)))
if __name__=='__main__':
    test_RFE()
    test_compare_with_no_feature_selection()
    test_RFECV() 