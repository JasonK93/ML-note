

from sklearn.svm import  LinearSVC
from sklearn.datasets import  load_digits
from sklearn import  cross_validation
from sklearn.linear_model import LogisticRegression
from  sklearn.pipeline import Pipeline
def test_Pipeline(data):
    '''
    test the pipeline
    :param data:  train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    steps=[("Linear_SVM",LinearSVC(C=1,penalty='l1',dual=False)),
           ("LogisticRegression",LogisticRegression(C=1))]
    pipeline=Pipeline(steps)
    pipeline.fit(X_train,y_train)
    print("Named steps:",pipeline.named_steps)
    print("Pipeline Score:",pipeline.score(X_test,y_test))
if __name__=='__main__':
    data=load_digits()
    test_Pipeline(cross_validation.train_test_split(data.data, data.target,test_size=0.25
			,random_state=0,stratify=data.target))