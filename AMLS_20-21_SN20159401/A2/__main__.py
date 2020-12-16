from sklearn.svm import SVC
from sklearn import metrics


class A2(object):
    def __init__(self):
        super().__init__()
        self.model = SVC(kernel='linear')

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_val)
        return metrics.accuracy_score(y_val, pred)

    def test(self, X, y):
        pred = self.model.predict(X)
        return metrics.accuracy_score(y, pred)
