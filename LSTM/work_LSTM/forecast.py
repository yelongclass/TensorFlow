#https://github.com/addfor/tutorials/tree/master/machine_learning
#ml16v04_forecasting_with_LSTM.ipynb

import numpy as np
import pandas as pd
#from neon import NervanaObject
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from IPython.display import Image

from bokeh.io import vplot, gridplot
import bokeh.plotting as bk

import tensorflow as tf
import tensorflow.contrib.learn as skflow

from lstm import lstm_model

bk.output_notebook()

data = pd.read_csv('data/data2.csv', parse_dates=['X0'])

# tools = 'pan,wheel_zoom,box_zoom,reset'
#
# fig_a = bk.figure(plot_width=800, plot_height=350,
#                   x_axis_label='time',
#                   y_axis_label='value',
#                   x_axis_type='datetime', tools=tools)
# fig_a.line(data['X0'], data['y'])
# bk.show(fig_a)

tools = 'pan,wheel_zoom,box_zoom,reset'

prediction = 12         #prediction steps
steps_forward =prediction
steps_backward = 0      #must be negtive or zero
inputs_default = 0
hidden = 128
batch_size = 256
n_steps = seq_len = 1       #sequence length,represents the number of elements that we would like to use when classifying a sequence
epochs = 100
test_sets = 6               #test number
test_size = 0.2             #test perchentage

TIMESTEPS = 139
RNN_LAYERS = [{'num_units': 139}, {'num_units': 69}]
DENSE_LAYERS = None #[10, 10]
LOG_DIR = './log'
# LEARNING_RATE = 0.001
# LEARNING_STEP = 100000
# PRINT_STEPS = 1000

#specify the number of steps forward we want added to the inputs datasets
#have a number of steps equal the number of steps in the future
# the inputs datasets
input_range = {
    'X109': [steps_backward, steps_forward],
    'X110': [steps_backward, steps_forward],
    'X111': [steps_backward, steps_forward],
    'X112': [steps_backward, steps_forward],
    'X70': [steps_backward, steps_forward],
    'X71': [steps_backward, steps_forward],
    'X73': [steps_backward, steps_forward],
    'X91': [steps_backward, steps_forward],
    'X92': [steps_backward, steps_forward],
    'X94': [steps_backward, steps_forward],
}

#choose the subset of columns that we want in the dataset, and the column that represent the target value.
X_columns = ['X109',  'X54', 'X53', 'X71', 'X112', 'X59', 'X111', 'X92', 'X66',
             'X94', 'X73', 'X91', 'X110', 'X40', 'y', 'X47', 'X48', 'X70', 'X60']
y_column = 'y'

#creates X and y
def gen_Xy(source, y_column, X_columns, inputs_per_column=None, inputs_default=3, steps_forward=1):
    #insert steps_foward NaN at the bottom
    y = source[y_column].shift(-steps_forward)
    #Normalized the X_columns
    scaler = StandardScaler()
    new_X = pd.DataFrame(scaler.fit_transform(source[X_columns]), columns=X_columns)
    X = pd.DataFrame()

    for column in X_columns:
        inputs = inputs_per_column.get(column, None)
        if inputs:
            inputs_list = range(inputs[0], inputs[1]+1)
        else:
            inputs_list = range(-inputs_default, 1)

        for i in inputs_list:
            col_name = "%s_%s" % (column, i)
            X[col_name] = new_X[column].shift(-i)   #Note: shift direction is inverted

    X = pd.concat([X, y], axis = 1)

    X.dropna(inplace = True, axis = 0)

    y = X[y_column].values.reshape(X.shape[0], 1)
    X.drop([y_column] , axis = 1, inplace = True)

    return X.values, y

#split the dataset in training and validation
def split_Xy(X, y, test_size, sets = 1):
    set_length = X.shape[0]/sets
    offset = 0
    X_train_lst = []
    y_train_lst = []
    X_test_lst = []
    y_test_lst = []
    for i in range(sets + 1):
        offset = i * set_length
        tr_length = int(set_length * (1-test_size))
        tr_s = offset                       #Train Start
        tr_e = offset + tr_length           #Train End
        te_s = tr_e                         #Test Start
        te_e = (i+1) * set_length           #Test End
        X_train_lst.append(X[tr_s:tr_e])
        y_train_lst.append(y[tr_s:tr_e])
        X_test_lst.append(X[te_s:te_e])
        y_test_lst.append(y[te_s:te_e])
    X_train = np.concatenate(X_train_lst)
    y_train = np.concatenate(y_train_lst)
    X_test = np.concatenate(X_test_lst)
    y_test = np.concatenate(y_test_lst)
    return X_train, y_train, X_test, y_test

X, y = gen_Xy(source=data, y_column='y', X_columns=X_columns,
                           inputs_per_column=input_range, inputs_default=inputs_default,
                           steps_forward=prediction)

X_train, y_train, X_test, y_test = split_Xy(X, y, test_size, test_sets)

#limits memory usage to avoid complete GPU allocation to TensorFlow
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

#build an LSTM Neural Network with a single layer and hidden number of neurons
def lstm_model(num_units, rnn_layers, dense_layers=None, learning_rate=0.1, optimizer='Adagrad'):
    """
    Creates a deep model based on:
        * stacked lstm cells
        * an optional dense layers
    :param num_units: the size of the cells.
    :param rnn_layers: list of int or dict
                         * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                         * list of dict: [{steps: int, keep_prob: int}, ...]
    :param dense_layers: list of nodes for each layer
    :return: the model definition
    """

    def lstm_cells(layers):
        if isinstance(layers[0], dict):
            return [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(layer['num_units'],
                                                                               state_is_tuple=True),
                                                  layer['keep_prob'])
                    if layer.get('keep_prob') else tf.nn.rnn_cell.BasicLSTMCell(layer['num_units'],
                                                                                state_is_tuple=True)
                    for layer in layers]
        return [tf.nn.rnn_cell.BasicLSTMCell(steps, state_is_tuple=True) for steps in layers]

    def dnn_layers(input_layers, layers):
        if layers and isinstance(layers, dict):
            return tflayers.stack(input_layers, tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            return input_layers

    def _lstm_model(X, y):
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        x_ = tf.unpack(X, axis=1, num=num_units)
        output, layers = tf.nn.rnn(stacked_lstm, x_, dtype=dtypes.float32)
        output = dnn_layers(output[-1], dense_layers)
        prediction, loss = tflearn.models.linear_regression(output, y)
        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_step(), optimizer=optimizer,
            learning_rate=learning_rate)
        return prediction, loss, train_op

    return _lstm_model

n_input = X_train.shape[1]
steps = (X_train.shape[0] / batch_size) * epochs       #steps is the number of batches to preprocessing

#convert all variables to type float32
X_train = X_train.astype(np.float32).copy()
y_train = y_train.astype(np.float32).copy()
X_test = X_test.astype(np.float32).copy()
y_test = y_test.astype(np.float32).copy()
X, y = X.astype(np.float32).copy(), y.astype(np.float32).copy()


#use it as a regression estimator you have to set n_classes to 0.
#model_params = {"learning_rate": LEARNING_RATE, "steps": steps, "batch_size": batch_size, "verbose": 1, "n_classes": 0}
# model = skflow.TensorFlowEstimator(model_fn=lstm_model, n_classes=0, verbose=1,
#                                  batch_size=batch_size, steps=steps)

#.SKCompat(skflow.Estimator(model_fn=lstm_model(X, y), model_dir=LOG_DIR, params = ))
model = skflow.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS), model_dir=LOG_DIR)

# create a lstm instance and validation monitor
validation_monitor = skflow.monitors.ValidationMonitor(X, y,
                                                     every_n_steps=steps,
                                                     early_stopping_rounds=1000)
#model.fit(X_train, y_train, logdir='tmp/')
model.fit(X_train, y_train, monitors=[validation_monitor], batch_size = batch_size, steps=steps)

y_train_predicted = model.predict(X_train)

trainScore = sqrt(mean_squared_error(y_train, y_train_predicted))
print('Train Score: %.2f RMSE' % (trainScore))
y_test_predicted = model.predict(X_test)
testScore = sqrt(mean_squared_error(y_test, y_test_predicted))
print('Test Score: %.2f RMSE' % (testScore))
y_hat = model.predict(X)

fig_b = bk.figure(plot_width=800, plot_height=350,
                  x_axis_label='time',
                  y_axis_label='value',
                  tools=tools)
fig_b.line(xrange(len(y)), np.ravel(y), legend='true value')
fig_b.line(xrange(len(y_hat)), np.ravel(y_hat), color='orange', legend='predicted value')
bk.show(fig_b)
