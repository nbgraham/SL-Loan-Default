class Wrapper():
    def __init__(self, my_code, my_model, sklearn_model):
        self.my_code = my_code
        self.model = my_model if self.my_code else sklearn_model

    def fit(self,x,y,attributes):
        return self.model.fit(x,y,attributes) if self.my_code else self.model.fit(x,y)

    def predict_prob(self,x):
        return self.model.predict_prob(x) if self.my_code else self.model.predict_proba(x)