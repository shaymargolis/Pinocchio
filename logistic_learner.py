from sklearn.linear_model import LogisticRegression

class LogisticLearner:
    def __init__(self):
        pass

    def learn(self, train_x, train_y):
        self.lm = LogisticRegression()
        self.lm.fit(train_x, train_y)

    def predict(self, test_x):
        return self.lm.predict(test_x)
