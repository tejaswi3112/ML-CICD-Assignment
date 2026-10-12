from model import train_model
from sklearn.datasets import load_iris


def test_model_training():
    model, accuracy = train_model()
    assert model is not None
    assert accuracy >= 0.90


def test_iris_dataset():
    iris = load_iris()
    assert iris.data.shape == (150, 4)
    assert len(iris.target) == 150


def test_model_prediction():
    model, _ = train_model()
    iris = load_iris()

    prediction = model.predict(iris.data[:5])

    assert len(prediction) == 5
    assert all(p in [0, 1, 2] for p in prediction)
```
