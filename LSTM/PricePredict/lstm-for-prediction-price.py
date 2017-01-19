from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lstm_predictor import generate_data, load_csvdata, lstm_model


LOG_DIR = './ops_logs/lstm_price'
TIMESTEPS = 10
RNN_LAYERS = [{'num_units': TIMESTEPS}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 100000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100

dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M')
rawdata = pd.read_csv("./data/RealMarketPriceDataPT.csv",
                   parse_dates={'timeline': ['date', '(UTC)']},
                   index_col='timeline', date_parser=dateparse)


X, y = load_csvdata(rawdata, TIMESTEPS, seperate=False)


regressor = learn.SKCompat(learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),
                                model_dir=LOG_DIR))

# create a lstm instance and validation monitor
validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                     every_n_steps=PRINT_STEPS,
                                                     early_stopping_rounds=1000)

regressor.fit(X['train'], y['train'],
              monitors=[validation_monitor],
#              batch_size=BATCH_SIZE,
              steps=TRAINING_STEPS)


'''
regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),
                                      n_classes=0,
                                      verbose=1,
                                      steps=TRAINING_STEPS,
                                      optimizer='Adagrad',
                                      learning_rate=0.03,
                                      batch_size=BATCH_SIZE)


regressor = tf.contrib.learn.DNNClassifier(feature_columns=[X, y],
                                       hidden_units=[10, 10, 10],
                                       activation_fn=tf.nn.relu,
                                       dropout=0.2,
                                       n_classes=3,
                                       optimizer="Adam")


validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                      every_n_steps=PRINT_STEPS,
                                                      early_stopping_rounds=1000)

regressor.fit(X['train'], y['train'], steps=TRAINING_STEPS, batch_size=BATCH_SIZE, monitors=[validation_monitor])
#regressor.fit(X['train'], y['train'], monitors=[validation_monitor], logdir=LOG_DIR)
'''

predicted = regressor.predict(X['test'])
mse = mean_absolute_error(y['test'], predicted)
print ("Error: %f" % mse)

plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(y['test'], label='test')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()
