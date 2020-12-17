from sklearn.svm import SVC
from sklearn import metrics


class B1(object):
    def __init__(self, use_CNN):
        super().__init__()
        self.use_CNN = use_CNN
        if use_CNN:
            # TODO
            pass
        else:
            self.model = SVC(kernel='poly', decision_function_shape='ovo')

    def train(self, train, val):
        if self.use_CNN:
            # TODO
            pass
        else:
            self.model.fit(train[0], train[1])
            pred = self.model.predict(val[0])
            return metrics.accuracy_score(val[1], pred)

    def test(self, test):
        if self.use_CNN:
            # TODO
            pass
        else:
            pred = self.model.predict(test[0])
            return metrics.accuracy_score(test[1], pred)


