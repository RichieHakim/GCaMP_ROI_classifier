
# Contact Josh (Joshua_Zimmer@hms.harvard.edu) for assistance.

import numpy as np
import tensorflow as tf
keras = tf.keras
layers = keras.layers
regularizers = keras.regularizers

def predict_channel_last(data: np.ndarray, model_name: str='resnet_1f'):
    '''
    Returns the predicted class probabilities for each batch example based on the model. (Channel is last dim)

            Parameters:
                    data (np.ndarray): Input image(s) to be classified (Batch Size x 36 x 36 x 1)
                    model_name (str): Folder name of model to be applied

            Returns:
                    pred (np.ndarray): Predicted predicted class probabilities for each example (Batch Size x 6)
    '''
    model = keras.models.load_model(f'./{model_name}')
    pred = model.predict(data)
    return pred


def predict(data: np.ndarray, model_name: str='resnet_1f'):
    '''
    Returns the predicted class probabilities for each batch example based on the model. (Channel is 2nd dim)

            Parameters:
                    data (np.ndarray): Input image(s) to be classified (Batch Size x 1 x 36 x 36)
                    model_name (str): Folder name of model to be applied

            Returns:
                    pred (np.ndarray): Predicted predicted class probabilities for each example (Batch Size x 6)
    '''
    data = np.transpose(data,(0,2,3,1))
    return predict_channel_last(data, model_name)

def get_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    '''
    Returns the prediction accuracies of y_pred against y_true via argmax.

            Parameters:
                    y_true (np.ndarray): Sparse prediction labels (i.e. 0, 1, 2, 3, 4, or 5) in Batch Size dims (i.e. 1D vector)
                    y_pred (np.ndarray): Prediction percentages in Batch Size x 6 dims

            Returns:
                    acc_val (float): Associated Accuracies between y_true and y_pred
    '''
    acc = keras.metrics.Accuracy()
    acc.update_state(np.argmax(y_pred, -1), y_true)
    acc_val = acc.result().numpy()
    return acc_val

def get_crossentropy(y_true: np.ndarray, y_pred: np.ndarray):
    '''
    Returns the cross_entropy of y_pred vs. y_true labels via SparseCategoricalCrossentropy

            Parameters:
                    y_true (np.ndarray): Sparse prediction labels (i.e. 0, 1, 2, 3, 4, or 5) in Batch Size dims (i.e. 1D vector)
                    y_pred (np.ndarray): Prediction percentages in Batch Size x 6 dims

            Returns:
                    sccpred (float): Associated Cross Entropy Score between y_true and y_pred
    '''
    scc = keras.losses.SparseCategoricalCrossentropy()
    sccpred = scc(y_true, y_pred).numpy()
    return sccpred

