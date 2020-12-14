from sklearn.svm import LinearSVC
from sklearn import metrics


class A1(object):
    def __init__(self):
        super().__init__()
        self.model = LinearSVC(penalty='l2', dual=False, C=1.0, class_weight='balanced')

    def train(self, X_train, y_train, X_val, y_val):
        print(X_train.shape, y_train.shape)
        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_val)
        return metrics.accuracy_score(y_val, pred)

    def test(self, X, y):
        pred = self.model.predict(X)
        return metrics.accuracy_score(y, pred)
