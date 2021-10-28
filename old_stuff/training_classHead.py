import torch
import numpy as np

class HeadModel():
    def __init__(self, model, SupervisedClass):
        '''

        '''
        self.model = model
        self.SupervisedClass = SupervisedClass
        
        self.is_trained = False
        self.n_classes = None
    
    def fit(self, X_train, y_train, **kwargs):
        '''
        Train a head regression model on the training set.

        JZ 2021
        
        Args:
            X_train: training features
            y_train: training labels
            **kwargs: keyword arguments to pass to the LogisticRegression class
        Returns: the trained model
        '''
        head_interim = self.get_simCLR_head(X_train)
        self.headmodel = self.SupervisedClass(**kwargs)
        self.headmodel.fit(head_interim, y_train)
        self.is_trained = True
        self.n_classes = np.max(y_train) + 1
        return self

    def predict_proba(self, X):
        '''
        Predict the label probabilities of the features using the model and the head model.

        JZ 2021
        
        Args:
            model: the trained model
            headmodel: the trained head model
            X: the features
        Returns: the predicted labels
        '''
        if self.is_trained:
            head_interim = self.get_simCLR_head(X)
            prediction = self.headmodel.predict_proba(head_interim)
        else:
            prediction = None
        return prediction

    def predict(self, X):
        '''
        Predict the labels of the features using the model and the head model.

        JZ 2021
        
        Args:
            model: the trained model
            headmodel: the trained head model
            X: the features
        Returns: the predicted labels
        '''
        if self.is_trained:
            head_interim = self.get_simCLR_head(X)
            prediction = self.headmodel.predict(head_interim)
        else:
            prediction = None
        return prediction

    def get_simCLR_head(self, X):
        '''
        Get the intermediate step of the model on the features.

        JZ 2021
        
        Args:
            model: the trained model
            X: the features
        Returns: the intermediate step of the model
        '''
        tensor_X = torch.tensor(X, dtype=torch.float)
        return self.model.cnn_layers(tensor_X).squeeze().detach().numpy()

    def get_simCLR_output(self, X):
        '''
        Get the output of the model on the features.

        JZ 2021

        Args:
            model: the trained model
            X: the features
        Returns: the output of the model
        '''
        tensor_X = torch.tensor(X, dtype=torch.float)
        return self.model(tensor_X).detach().numpy()
    
    def score(self, X, y):
        '''
        Score the model on the features and the labels.

        JZ 2021
        
        Args:
            model: the trained model
            X: the features
            y: the labels
        Returns: the score of the model
        '''
        return self.headmodel.score(self.get_simCLR_head(X), y)






# def train_head(model, SupervisedClass, X_train, y_train, **kwargs):
#     '''
#     Train a head regression model on the training set.

#     JZ 2021
    
#     Args:
#         X_train: training features
#         y_train: training labels
#         **kwargs: keyword arguments to pass to the LogisticRegression class
#     Returns: the trained model
#     '''
#     head_interim = get_simCLR_head(model, X_train)
#     head_model = SupervisedClass(**kwargs)
#     head_model.fit(head_interim, y_train)
#     return head_model

# def predict_proba(model, headmodel, X):
#     '''
#     Predict the label probabilities of the features using the model and the head model.

#     JZ 2021
    
#     Args:
#         model: the trained model
#         headmodel: the trained head model
#         X: the features
#     Returns: the predicted labels
#     '''
#     head_interim = get_simCLR_head(model, X)
#     return headmodel.predict_proba(head_interim)

# def predict(model, headmodel, X):
#     '''
#     Predict the labels of the features using the model and the head model.

#     JZ 2021
    
#     Args:
#         model: the trained model
#         headmodel: the trained head model
#         X: the features
#     Returns: the predicted labels
#     '''
#     head_interim = get_simCLR_head(model, X)
#     return headmodel.predict(head_interim)

# def get_simCLR_head(model, X):
#     '''
#     Get the intermediate step of the model on the features.

#     JZ 2021
    
#     Args:
#         model: the trained model
#         X: the features
#     Returns: the intermediate step of the model
#     '''
#     tensor_X = torch.tensor(X, dtype=torch.float)
#     return model.cnn_layers(tensor_X).squeeze().detach().numpy()

# def get_simCLR_output(model, X):
#     '''
#     Get the output of the model on the features.

#     JZ 2021

#     Args:
#         model: the trained model
#         X: the features
#     Returns: the output of the model
#     '''
#     tensor_X = torch.tensor(X, dtype=torch.float)
#     return model(tensor_X).detach().numpy()
