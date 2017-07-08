import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.learning_curve import validation_curve

def test_validation_curve():
    '''
    test validation_curve with LinerSVC and different C
    :return:  None
    '''
    digits = load_digits()
    X,y=digits.data,digits.target
    param_name="C"
    param_range = np.logspace(-2, 2)
    train_scores, test_scores = validation_curve(LinearSVC(), X, y, param_name=param_name,
             param_range=param_range,cv=10, scoring="accuracy")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)

    ax.semilogx(param_range, train_scores_mean, label="Training Accuracy", color="r")
    ax.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    ax.semilogx(param_range, test_scores_mean, label="Testing Accuracy", color="g")
    ax.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")

    ax.set_title("Validation Curve with LinearSVC")
    ax.set_xlabel("C")
    ax.set_ylabel("Score")
    ax.set_ylim(0,1.1)
    ax.legend(loc='best')
    plt.show()

if __name__=='__main__':
    test_validation_curve()