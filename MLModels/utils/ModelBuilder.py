import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.optimizers
from tensorflow.keras import backend as K
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Create the /tmp directory if it doesn't already exist
import os
if not os.path.exists('tmp'):
    os.makedirs('tmp')


def rmse (y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred -y_true)))


# Construct a learning rate scheduler such that the lr is decreased when the loss remains unchanged after 4 epochs
# Adapted from
# https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self, minimum_learning_rate):
        self.losses = []
        self.lr = []
        self.last_loss_improvement = None
        self.loss_unchanged_count = 0
        self.minimum_learning_rate = minimum_learning_rate

    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        self.last_loss_improvement = 1.0
        self.loss_unchanged_count = 0

    def on_epoch_end(self, batch, logs={}):
        epoch_loss = logs.get('val_loss')
        # track if losses are being updated
        if (len(self.losses) > 1) and ((epoch_loss == self.losses[-1]) or (epoch_loss > self.last_loss_improvement)):
            self.loss_unchanged_count += 1
        else:
            self.loss_unchanged_count = 0
            self.last_loss_improvement = epoch_loss

        self.losses.append(epoch_loss)

    # Implement the lr_loss_scheduler within the class so that it can access loss_unchanged_count
    def on_epoch_begin(self, batch,
                       logs={}):
        current_lr = K.get_value(self.model.optimizer.lr)
        lr = current_lr - (current_lr / 4)  # propose to decrement lr by 25% of current value
        if self.loss_unchanged_count > 3 and lr >= self.minimum_learning_rate:
            self.loss_unchanged_count = 0  # reset
            K.set_value(self.model.optimizer.lr, lr)  # set for the model
            print('Updated learning rate to ', lr)
            # return K.get_value(self.model.optimizer.lr)


metric_descriptions = {'acc': ['Accuracy', 'Accuracy'],
                           'accuracy': ['Accuracy', 'Accuracy'],
                           'auc': ['AUC', 'Area Under the Curve'],
                           'rmse': ['RMSE', 'Root Mean Squared Error'],
                           'loss': ['Loss', 'Loss'],
                           'mae': ['MAE', 'Mean Absolute Error'],
                           'msle': ['MSLE', 'Mean Squared Logarithmic Error'],
                           'mse': ['MSE', 'Mean Squared Error'],
                           'poisson': ['Poisson Loss', 'Poisson Loss']
                       }


# Function to display important model metrics. Namely, Accuracy, Root Mean Square Error and loss
# accuracy key for the fitted model history is different for Kaggle environment
def print_model_metrics(model_history, acc_key, metrics=None):

    if metrics is None: # if the desired model metrics are not explicitly defined
        metrics = [key for key in model_history.history if not key.startswith('val_')]

    print('Diagrammed History of Model Metrics')

    for metric in metrics:
        plt.plot(model_history.history[metric], color='black')
        plt.plot(model_history.history['val_' + metric], color='blue')
        name, long_name = (metric_descriptions[metric][0], metric_descriptions[metric][1]) \
            if metric in metric_descriptions else (metric.capitalize(), metric.capitalize())

        plt.title('Model {} over Epochs'.format(long_name))
        plt.ylabel(name)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()


def print_model_metrics_multiple_iterations(history_list, acc_key, metrics=None):
    if metrics is None: # if the desired model metrics are not explicitly defined
        metrics = [key for key in history_list[0].history if not key.startswith('val_')]

    print('Diagrammed History of Model Metrics')

    for metric in metrics:
        for h in history_list:
            plt.plot(h.history[metric], color='black')
            plt.plot(h.history['val_' + metric], color='blue')
        name, long_name = (metric_descriptions[metric][0], metric_descriptions[metric][1]) \
            if metric in metric_descriptions else (metric.capitalize(), metric.capitalize())

        plt.title('{} of all Models over Epochs'.format(long_name))
        plt.ylabel(name)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()


def construct_and_run_model(data, model, optimizer, checkpoint_file_name, epochs=150,
                            batch_size=64, loss_type='categorical_crossentropy',
                            metrics=['accuracy', rmse], minimum_learning_rate=10.e-6,
                            verbose_fit=2, verbose_checkpointer=1, additional_callbacks=None,
                            plot_confusion_matrix=True):
    loss_history = LossHistory(minimum_learning_rate)

    # compile the model and set the optimizers
    model.compile(loss=loss_type, optimizer=optimizer,
                  metrics=metrics)

    model.summary()

    # count number of parameters in the model
    numParams = model.count_params()

    # set a valid path to record model checkpoints
    checkpointer = ModelCheckpoint(filepath=checkpoint_file_name, verbose=verbose_checkpointer,
                                   save_best_only=True)
    callbacks=[checkpointer, loss_history]
    if isinstance(additional_callbacks, list):
         callbacks.extend(additional_callbacks)
    # change_lr = tensorflow.keras.callbacks.LearningRateScheduler(lr_loss_scheduler)
    print('MB::   Shape of X_train: ', np.asarray(data['X_train']).shape)
    print('MB::   Shape of X_validate: ', np.asarray(data['X_validate']).shape)
    print('MB::   Shape of X_test: ', np.asarray(data['X_test']).shape)
    history = model.fit(data['X_train'], data['Y_train'], batch_size=batch_size, epochs=epochs,
                             verbose=verbose_fit, validation_data=(data['X_validate'], data['Y_validate']),
                             callbacks=callbacks)

    model.load_weights(checkpoint_file_name)

    probs = model.predict(data['X_test'])
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == data['Y_test'].argmax(axis=-1))
    results = model.evaluate(data['X_test'], data['Y_test'], batch_size=32)
    print("\nTest classification accuracy using model.predict : %f " % acc)
    print('\nTest metrics: ')
    for i, r in enumerate(results):
        print(list(history.history.keys())[i].capitalize(), ': ', r)

    if plot_confusion_matrix:
        print('\nPlot of confusion matrix')
        y_test = [list(label).index(1.) for label in data['Y_test']]  # revert to list of labels
        cm = confusion_matrix(y_test, preds)
        df_cm = pd.DataFrame(cm)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True)
        plt.pause(0.05)
        plt.show()

    return history


def construct_and_run_model_loo(data, model, optimizer, checkpoint_file_name, epochs=150,
                                batch_size=64, loss_type='categorical_crossentropy',
                                metrics=['accuracy', rmse], minimum_learning_rate=10.e-6,
                                verbose_fit=2, verbose_checkpointer=1, additional_callbacks=None,
                                plot_confusion_matrix=True):
    print('Deprecated function. Use construct_and_run_model_multiple_iterations() instead.')
    model, test_metrics, history \
        = construct_and_run_model_multiple_iterations(data, model, optimizer,
                                                      checkpoint_file_name, epochs=epochs,
                                                      batch_size=batch_size, loss_type=loss_type, metrics=metrics,
                                                      minimum_learning_rate=minimum_learning_rate,
                                                      verbose_fit=verbose_fit,
                                                      verbose_checkpointer=verbose_checkpointer,
                                                      additional_callbacks=additional_callbacks,
                                                      plot_confusion_matrix=plot_confusion_matrix)
    return model, test_metrics, history


# Compiles and runs pre-defined model. Returns the fitted model, test metrics, and training/validation history
def construct_and_run_model_multiple_iterations(data, model, optimizer, checkpoint_file_name, epochs=150,
                                batch_size=64, loss_type='categorical_crossentropy',
                                metrics=['accuracy', rmse], minimum_learning_rate=10.e-6,
                                verbose_fit=2, verbose_checkpointer=1, additional_callbacks=None,
                                plot_confusion_matrix=True):

    loss_history = LossHistory(minimum_learning_rate)

    # compile the model and set the optimizers
    model.compile(loss=loss_type, optimizer=optimizer,
                  metrics=metrics)

    model.summary()

    # count number of parameters in the model
    numParams = model.count_params()

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath=checkpoint_file_name, verbose=verbose_checkpointer,
                                   save_best_only=True)

    callbacks = [checkpointer, loss_history]
    if isinstance(additional_callbacks, list):
        callbacks.extend(additional_callbacks)

    # change_lr = tensorflow.keras.callbacks.LearningRateScheduler(lr_loss_scheduler)
    print('MB::   Shape of X_train: ', np.asarray(data['X_train']).shape)
    print('MB::   Shape of X_validate: ', np.asarray(data['X_validate']).shape)
    print('MB::   Shape of X_test: ', np.asarray(data['X_test']).shape)
    history = model.fit(data['X_train'], data['Y_train'], batch_size=batch_size, epochs=epochs,
                        verbose=verbose_fit, validation_data=(data['X_validate'], data['Y_validate']),
                        callbacks=callbacks)

    model.load_weights(checkpoint_file_name)

    probs = model.predict(data['X_test'])
    preds = probs.argmax(axis=-1)
    print('dbg: preds: ', preds)
    acc = np.mean(preds == data['Y_test'].argmax(axis=-1))
    results = model.evaluate(data['X_test'], data['Y_test'], batch_size=32)
    print("\nTest classification accuracy using model.predict : %f " % acc)
    print('\nTest metrics: ')
    for i, r in enumerate(results):
        print(list(history.history.keys())[i].capitalize(), ': ', r)

    if plot_confusion_matrix:
        print('\nPlot of confusion matrix')
        y_test = [list(label).index(1.) for label in data['Y_test']]  # revert to list of labels
        cm = confusion_matrix(y_test, preds)
        df_cm = pd.DataFrame(cm)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True)
        plt.pause(0.05)

        plt.show()

    return model, results, history


