import torch

def train_head(HeadClass, model, X_train, y_train, **kwargs):
    '''
    Train a head regression model on the training set.

    JZ 2021
    
    Args:
        X_train: training features
        y_train: training labels
        **kwargs: keyword arguments to pass to the LogisticRegression class
    Returns: the trained model
    '''
    head_interim = get_simCLR_interim(model, X_train)
    logreg = HeadClass(**kwargs)
    logreg.fit(head_interim, y_train)
    return logreg

def predict_head_proba(model, headmodel, X):
    '''
    Predict the label probabilities of the features using the model and the head model.

    JZ 2021
    
    Args:
        model: the trained model
        headmodel: the trained head model
        X: the features
    Returns: the predicted labels
    '''
    head_interim = get_simCLR_interim(model, X)
    return headmodel.predict_proba(head_interim)

def predict_head(model, headmodel, X):
    '''
    Predict the labels of the features using the model and the head model.

    JZ 2021
    
    Args:
        model: the trained model
        headmodel: the trained head model
        X: the features
    Returns: the predicted labels
    '''
    head_interim = get_simCLR_interim(model, X)
    return headmodel.predict(head_interim)

def get_simCLR_interim(model, X):
    '''
    Get the intermediate step of the model on the features.

    JZ 2021
    
    Args:
        model: the trained model
        X: the features
    Returns: the intermediate step of the model
    '''
    tensor_X = torch.tensor(X, dtype=torch.float)
    return model.cnn_layers(tensor_X).squeeze().detach().numpy()

def get_simCLR_output(model, X):
    '''
    Get the output of the model on the features.

    JZ 2021

    Args:
        model: the trained model
        X: the features
    Returns: the output of the model
    '''
    tensor_X = torch.tensor(X, dtype=torch.float)
    return model(tensor_X).detach().numpy()
