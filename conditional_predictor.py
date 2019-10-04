from __future__ import print_function
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from mnist_model import get_untrained_model, get_dataset

(x_train, y_train), (x_test, y_test), input_shape = get_dataset()
batch_size = 128
epochs = 2

model = get_untrained_model(num_filter_1=4, input_shape=input_shape)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1
          )


class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        print("Transformer.fit: ", X.shape)
        return self

    def transform(self, X, copy=None):
        print("Transformer.transform: ", X.shape)
        return self.model.predict(X)


class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        print("Classifier.fit: ", X.shape)
        assert (type(self.threshold) == float), "threshold parameter must be float"
        self._threshold = self.threshold
        return self

    def predict(self, X, y=None):
        print("Classifier.predict: ", X.shape)
        try:
            getattr(self, "_threshold")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        X[X < self._threshold] = 0
        return X

    def score(self, X, y=None, sample_weight=None):
        # counts number of values bigger than mean
        return self.predict(X)


def get_conditional_prob(model, threshold):
    steps = [('transformer', Transformer(model=model)),
             ('classifier', Classifier(threshold=threshold))]
    pipeline = Pipeline(steps)
    pipeline.fit(x_train, y_train)
    return pipeline.predict(x_test)


y_pred = get_conditional_prob(model=model, threshold=0.80)
print("Output Shape: ", y_pred.shape)