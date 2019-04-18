import joblib
from keras import models


def save_sklearn_model(model, filename):
    """
    Saves a Scikit-learn model to disk.
    Example of usage:
    
    >>> reg = sklearn.linear_models.LinearRegression()
    >>> reg.fit(x_train, y_train)
    >>> save_sklearn_model(reg, 'my_model.pickle')    

    :param model: the model to save;
    :param filename: string, path to the file in which to store the model.
    :return: the model.
    """
    joblib.dump(model, filename)


def load_sklearn_model(filename):
    """
    Loads a Scikit-learn model saved with joblib.dump.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = joblib.load(filename)

    return model


def save_keras_model(model, filename):
    """
    Saves a Keras model to disk.
    Example of usage:

    >>> model = Sequential()
    >>> model.add(Dense(...))
    >>> model.compile(...)
    >>> model.fit(...)
    >>> save_keras_model(model, 'my_model')

    :param model: the model to save;
    :param filename: string, path to the file in which to store the model.
    :return: the model.
    """
    models.save_model(model, filename)


def load_keras_model(filename):
    """
    Loads a compiled Keras model saved with models.save_model.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = models.load_model(filename)

    return model

