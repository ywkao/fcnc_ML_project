#import keras
#import keras_layer_normalization

import tensorflow as tf
try:
    import tensorflow.keras as keras
except:
    import keras

def standard(max_objects, n_features):
  input_objects = keras.layers.Input(shape=(max_objects, n_features), name = 'input')
  
  # 1x1 convolutional layers
  dropout_rate = 0.10
  #conv = keras.layers.Convolution1D(64, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_1')(input_objects)
  #conv = keras.layers.Dropout(rate = dropout_rate, name = 'conv_dropout_1')(conv)
  #conv = keras.layers.Convolution1D(32, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_2')(conv)
  #conv = keras.layers.Dropout(rate = dropout_rate, name = 'conv_dropout_2')(conv)
  #conv = keras.layers.Convolution1D(16, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_3')(conv)
  #conv = keras.layers.Dropout(rate = dropout_rate, name = 'conv_dropout_3')(conv)
  #conv = keras.layers.Convolution1D(6, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', name = 'conv_4')(conv)
  #conv = keras.layers.Dropout(rate = dropout_rate, name = 'conv_dropout_4')(conv)

  # lstm
  conv = input_objects
  lstm = keras.layers.LSTM(150, implementation=2, name='lstm_1', go_backwards=False)(conv)
  lstm = keras.layers.Dropout(rate = dropout_rate, name = 'lstm_dropout_1')(lstm)
  #lstm = keras.layers.LSTM(150, implementation=2, name='lstm_2', go_backwards=False)(lstm)
  #lstm = keras.layers.Dropout(rate = dropout_rate, name = 'lstm_dropout_2')(lstm)

  # fully-connected layers
  dropout_rate = 0.10
  dense = keras.layers.Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'dense_1')(lstm)
  dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_1')(dense)
  dense = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'dense_2')(dense)
  dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_2')(dense)
  dense = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'dense_3')(dense)
  dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_3')(dense)
  dense = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'dense_4')(dense)
  dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_4')(dense)
  dense = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', name = 'dense_5')(dense)

  # output
  output = keras.layers.Dense(1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = 'output')(dense)
  optimizer = keras.optimizers.Adam()

  model = keras.models.Model(inputs = [input_objects], outputs = [output])
  model.compile(optimizer = optimizer, loss = 'binary_crossentropy')
  print((model.summary()))

  return model


def baseline_leptonic_v1(max_objects, n_features, max_leptons, n_lepton_features, n_global_features, no_global, no_lstm, n_lstm):
  input_objects = keras.layers.Input(shape=(max_objects, n_features), name = 'input_objects')
  input_leptons = keras.layers.Input(shape=(max_leptons, n_lepton_features), name = 'input_leptons')
  input_global = keras.layers.Input(shape=(n_global_features,), name = 'input_global')

  # lstm
  maxnorm = 3
  batch_momentum = 0.6
  recurrent_dropout = 0.0
  nonrecurrent_dropout = 0.0
  lstm = keras.layers.LSTM(n_lstm, dropout = nonrecurrent_dropout, recurrent_dropout = recurrent_dropout, implementation=2, activation = 'tanh',  kernel_constraint = keras.constraints.max_norm(maxnorm), recurrent_constraint = keras.constraints.max_norm(maxnorm), name='lstm_1', go_backwards=False, return_sequences=True)(input_objects)
  #lstm = keras.layers.BatchNormalization(momentum = batch_momentum, name = "lstm_batchnorm_1")(lstm)
  #lstm = keras.layers.Dropout(rate = dropout_rate, name = 'lstm_dropout_1')(lstm)
  lstm = keras.layers.LSTM(n_lstm, dropout = nonrecurrent_dropout, recurrent_dropout = recurrent_dropout, implementation=2, activation = 'tanh',  kernel_constraint = keras.constraints.max_norm(maxnorm), recurrent_constraint = keras.constraints.max_norm(maxnorm), name='lstm_2', go_backwards=False, return_sequences=True)(lstm)
  #lstm = keras.layers.BatchNormalization(momentum = batch_momentum, name = "lstm_batchnorm_2")(lstm)
  #lstm = keras.layers.Dropout(rate = dropout_rate, name = 'lstm_dropout_2')(lstm)
  #lstm = keras.layers.LSTM(n_lstm, dropout = 0.0, recurrent_dropout = 0.0, implementation=2, activation = 'tanh',  kernel_constraint = keras.constraints.max_norm(maxnorm), recurrent_constraint = keras.constraints.max_norm(maxnorm), name='lstm_3', go_backwards=False, return_sequences=True)(lstm)
  #lstm = keras.layers.BatchNormalization(momentum = batch_momentum, name = "lstm_batchnorm_3")(lstm)
  #lstm = keras.layers.Dropout(rate = dropout_rate, name = 'lstm_dropout_3')(lstm)
  lstm = keras.layers.LSTM(n_lstm, dropout = nonrecurrent_dropout, recurrent_dropout = recurrent_dropout, implementation=2, activation = 'tanh',  kernel_constraint = keras.constraints.max_norm(maxnorm), recurrent_constraint = keras.constraints.max_norm(maxnorm), name='lstm_4', go_backwards=False)(lstm)
  #lstm = keras.layers.BatchNormalization(momentum = batch_momentum, name = "lstm_batchnorm_4")(lstm)
  #lstm = keras.layers.Dropout(rate = dropout_rate, name = 'lstm_dropout_4')(lstm)

  # lepton lstm
  lstm_lepton = keras.layers.LSTM(n_lstm, dropout = nonrecurrent_dropout, recurrent_dropout = recurrent_dropout, implementation=2, activation = 'tanh',  kernel_constraint = keras.constraints.max_norm(maxnorm), recurrent_constraint = keras.constraints.max_norm(maxnorm), name='lstm_lepton_1', go_backwards=False, return_sequences=True)(input_leptons)
  #lstm_lepton = keras.layers.BatchNormalization(momentum = batch_momentum, name = "lstm_lepton_batchnorm_1")(lstm)
  #lstm_lepton = keras.layers.Dropout(rate = dropout_rate, name = 'lstm_lepton_dropout_1')(lstm)
  lstm_lepton = keras.layers.LSTM(n_lstm, dropout = nonrecurrent_dropout, recurrent_dropout = recurrent_dropout, implementation=2, activation = 'tanh',  kernel_constraint = keras.constraints.max_norm(maxnorm), recurrent_constraint = keras.constraints.max_norm(maxnorm), name='lstm_lepton_2', go_backwards=False, return_sequences=True)(lstm_lepton)
  #lstm_lepton = keras.layers.BatchNormalization(momentum = batch_momentum, name = "lstm_lepton_batchnorm_2")(lstm)
  #lstm_lepton = keras.layers.Dropout(rate = dropout_rate, name = 'lstm_lepton_dropout_2')(lstm)
  #lstm_lepton = keras.layers.LSTM(n_lstm, dropout = 0.0, recurrent_dropout = 0.0, implementation=2, activation = 'tanh',  kernel_constraint = keras.constraints.max_norm(maxnorm), recurrent_constraint = keras.constraints.max_norm(maxnorm), name='lstm_lepton_3', go_backwards=False, return_sequences=True)(lstm)
  #lstm_lepton = keras.layers.BatchNormalization(momentum = batch_momentum, name = "lstm_lepton_batchnorm_3")(lstm)
  #lstm_lepton = keras.layers.Dropout(rate = dropout_rate, name = 'lstm_lepton_dropout_3')(lstm)
  lstm_lepton = keras.layers.LSTM(n_lstm, dropout = nonrecurrent_dropout, recurrent_dropout = recurrent_dropout, implementation=2, activation = 'tanh',  kernel_constraint = keras.constraints.max_norm(maxnorm), recurrent_constraint = keras.constraints.max_norm(maxnorm), name='lstm_lepton_4', go_backwards=False)(lstm_lepton)



  if no_global:
    print("Using LSTM only")
    merged_features = lstm
  elif no_lstm:
    print("Using global only")
    merged_features = input_global
  else:
    print("Using LSTM + global")
    merged_features = keras.layers.concatenate([input_global, lstm, lstm_lepton])

  # fully-connected layers
  maxnorm = 3
  dropout_rate = 0.1
  dense = keras.layers.Dense(300, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_1')(merged_features)
  dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_1')(dense)
  dense = keras.layers.Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_2')(dense)
  dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_2')(dense)
  dense = keras.layers.Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_3')(dense)
  #dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_3')(dense)
  #dense = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_4')(dense)
  #dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_4')(dense)
  #dense = keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_5')(dense)
  #dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_5')(dense)

  # output
  output = keras.layers.Dense(1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'output')(dense)
  optimizer = keras.optimizers.Adam()

  model = keras.models.Model(inputs = [input_global, input_objects, input_leptons], outputs = [output])
  model.compile(optimizer = optimizer, loss = 'binary_crossentropy')
  print((model.summary()))

  return model

def tth_learner(max_objects, n_features, n_global_features, config):
  input_objects = keras.layers.Input(shape=(max_objects, n_features), name = 'input_objects')
  input_global = keras.layers.Input(shape=(n_global_features,), name = 'input_global')

  # Hyperparameters
  maxnorm = config["maxnorm"]
  dropout_rate = config["dropout_rate"]

  n_nodes_lstm = config["n_nodes_lstm"]
  n_lstm = config["n_lstm"]

  n_nodes_dense_1 = config["n_nodes_dense_1"]
  n_dense_1 = config["n_dense_1"]
  n_nodes_dense_2 = config["n_nodes_dense_2"]
  n_dense_2 = config["n_dense_2"]

  learning_rate = config["learning_rate"]

  if "batch_norm" in list(config.keys()):
    batch_norm = config["batch_norm"]
    batch_momentum = config["batch_momentum"]
  else:
    batch_norm = False
  
  if "layer_norm" in list(config.keys()):
    layer_norm = config["layer_norm"]
  else:
    layer_norm = False

  if "epsilon" in list(config.keys()):
    epsilon = config["epsilon"]
  else:
    epsilon = 1e-08

  # LSTM
  lstm = input_objects

  for i in range(n_lstm):
    decay = 0
    lstm =  keras.layers.LSTM(n_nodes_lstm, implementation=2, activation='tanh', kernel_regularizer=keras.regularizers.l2(decay), recurrent_regularizer=keras.regularizers.l2(decay), bias_regularizer=keras.regularizers.l2(decay), kernel_constraint = keras.constraints.max_norm(maxnorm), go_backwards=False, return_sequences=(i != (n_lstm-1)), name='lstm_%d' % i)(lstm)
    if layer_norm: 
      #lstm = keras.layers.BatchNormalization(momentum = batch_momentum, name = 'lstm_batch_norm_%d' % i)(lstm)
      lstm = keras_layer_normalization.LayerNormalization()(lstm)

  # Merge LSTM output with high-level features in fully-connected layers
  dense = keras.layers.concatenate([input_global, lstm])
  for i in range(n_dense_1):
    dense = keras.layers.Dense(n_nodes_dense_1, activation = 'elu', kernel_initializer = 'lecun_uniform', kernel_regularizer = keras.regularizers.l2(decay), kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense1_%d' % i)(dense)
    if batch_norm:
      dense = keras.layers.BatchNormalization(momentum = batch_momentum, name = 'dense_batch_norm1_%d' % i)(dense)
    dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout1_%d' % i)(dense)

  for i in range(n_dense_2):
    dense = keras.layers.Dense(n_nodes_dense_2, activation = 'elu', kernel_initializer = 'lecun_uniform', kernel_regularizer = keras.regularizers.l2(decay), kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense2_%d' % i)(dense)
    if batch_norm and i < (n_dense_2 - 1):
      dense = keras.layers.BatchNormalization(momentum = batch_momentum, name = 'dense_batch_norm2_%d' % i)(dense) 
    if i < (n_dense_2 - 1):
      dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout2_%d' % i)(dense)

  output = keras.layers.Dense(1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'output')(dense)
  #optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = epsilon)
  #gradients, variables = zip(*optimizer.compute_gradients(loss))
  #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
  #optimize = optimizer.apply_gradients(zip(gradients, variables))
  #original_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = epsilon)
  #optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=1.0)

  optimizer = keras.optimizers.Adam(lr = learning_rate)

  model = keras.models.Model(inputs = [input_global, input_objects], outputs = [output])
  model.compile(optimizer = optimizer, loss = 'binary_crossentropy')
  print((model.summary()))
  print(("DNN Config:", config))

  return model


def baseline_v1(max_objects, n_features, n_global_features, no_global, no_lstm, n_lstm):
  input_objects = keras.layers.Input(shape=(max_objects, n_features), name = 'input_objects')
  input_global = keras.layers.Input(shape=(n_global_features,), name = 'input_global')

  maxnorm = 3
  # 1x1 convolutional layers
  dropout_rate = 0.10
  do_conv = False
  if do_conv:
    conv = keras.layers.Convolution1D(64, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'conv_1')(input_objects)
    conv = keras.layers.Dropout(rate = dropout_rate, name = 'conv_dropout_1')(conv)
    conv = keras.layers.Convolution1D(64, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'conv_2')(conv)
    conv = keras.layers.Dropout(rate = dropout_rate, name = 'conv_dropout_2')(conv)
    conv = keras.layers.Convolution1D(32, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'conv_3')(conv)
    #conv = keras.layers.Dropout(rate = dropout_rate, name = 'conv_dropout_3')(conv)
    conv = keras.layers.Convolution1D(5, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'conv_4')(conv)
    #conv = keras.layers.Dropout(rate = dropout_rate, name = 'conv_dropout_4')(conv)

  else:
    conv = input_objects

  # lstm
  batch_momentum = 0.6
  recurrent_dropout = 0.0
  nonrecurrent_dropout = 0.0
  lstm = keras.layers.LSTM(n_lstm, dropout = nonrecurrent_dropout, recurrent_dropout = recurrent_dropout, implementation=2, activation = 'tanh',  kernel_constraint = keras.constraints.max_norm(maxnorm), recurrent_constraint = keras.constraints.max_norm(maxnorm), name='lstm_1', go_backwards=False, return_sequences=True)(conv)
  #lstm = keras.layers.BatchNormalization(momentum = batch_momentum, name = "lstm_batchnorm_1")(lstm)
  #lstm = keras.layers.Dropout(rate = dropout_rate, name = 'lstm_dropout_1')(lstm)
  lstm = keras.layers.LSTM(n_lstm, dropout = nonrecurrent_dropout, recurrent_dropout = recurrent_dropout, implementation=2, activation = 'tanh',  kernel_constraint = keras.constraints.max_norm(maxnorm), recurrent_constraint = keras.constraints.max_norm(maxnorm), name='lstm_2', go_backwards=False, return_sequences=True)(lstm)
  #lstm = keras.layers.BatchNormalization(momentum = batch_momentum, name = "lstm_batchnorm_2")(lstm)
  #lstm = keras.layers.Dropout(rate = dropout_rate, name = 'lstm_dropout_2')(lstm)
  #lstm = keras.layers.LSTM(n_lstm, dropout = 0.0, recurrent_dropout = 0.0, implementation=2, activation = 'tanh',  kernel_constraint = keras.constraints.max_norm(maxnorm), recurrent_constraint = keras.constraints.max_norm(maxnorm), name='lstm_3', go_backwards=False, return_sequences=True)(lstm)
  #lstm = keras.layers.BatchNormalization(momentum = batch_momentum, name = "lstm_batchnorm_3")(lstm)
  #lstm = keras.layers.Dropout(rate = dropout_rate, name = 'lstm_dropout_3')(lstm)
  lstm = keras.layers.LSTM(n_lstm, dropout = nonrecurrent_dropout, recurrent_dropout = recurrent_dropout, implementation=2, activation = 'tanh',  kernel_constraint = keras.constraints.max_norm(maxnorm), recurrent_constraint = keras.constraints.max_norm(maxnorm), name='lstm_4', go_backwards=False)(lstm)
  #lstm = keras.layers.BatchNormalization(momentum = batch_momentum, name = "lstm_batchnorm_4")(lstm)
  #lstm = keras.layers.Dropout(rate = dropout_rate, name = 'lstm_dropout_4')(lstm)

  if no_global:
    print("Using LSTM only")
    merged_features = lstm
  elif no_lstm:
    print("Using global only")
    merged_features = input_global
  else:
    print("Using LSTM + global")
    merged_features = keras.layers.concatenate([input_global, lstm])

  # fully-connected layers
  maxnorm = 3
  dropout_rate = 0.25
  dense = keras.layers.Dense(300, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_1')(merged_features)
  dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_1')(dense)
  dense = keras.layers.Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_2')(dense)
  dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_2')(dense)
  dense = keras.layers.Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_3')(dense)
  dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_3')(dense)
  dense = keras.layers.Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_4')(dense)
  dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_4')(dense)
  dense = keras.layers.Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_5')(dense)
  #dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_5')(dense)

  # output
  output = keras.layers.Dense(1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'output')(dense)
  optimizer = keras.optimizers.Adam()

  model = keras.models.Model(inputs = [input_global, input_objects], outputs = [output])
  model.compile(optimizer = optimizer, loss = 'binary_crossentropy')
  print((model.summary()))

  return model

def cnn_v1(max_objects, n_features, n_global_features, no_global, no_lstm):
  input_objects = keras.layers.Input(shape=(max_objects, n_features), name = 'input_objects')
  input_global = keras.layers.Input(shape=(n_global_features,), name = 'input_global')

  maxnorm = 2
  # 1x1 convolutional layers
  dropout_rate = 0.15
  conv = keras.layers.Convolution1D(128, 2, kernel_initializer = 'lecun_uniform', activation = 'relu', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'conv_1')(input_objects)
  conv = keras.layers.Dropout(rate = dropout_rate, name = 'conv_dropout_1')(conv)
  conv = keras.layers.Convolution1D(128, 2, kernel_initializer = 'lecun_uniform', activation = 'relu', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'conv_2')(conv)
  conv = keras.layers.Dropout(rate = dropout_rate, name = 'conv_dropout_2')(conv)
  conv = keras.layers.Convolution1D(128, 2, kernel_initializer = 'lecun_uniform', activation = 'relu', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'conv_3')(conv)
  conv = keras.layers.Dropout(rate = dropout_rate, name = 'conv_dropout_3')(conv)
  conv = keras.layers.Convolution1D(128, 2, kernel_initializer = 'lecun_uniform', activation = 'relu', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'conv_4')(conv)
  conv = keras.layers.Flatten()(conv)

  if no_global:
    print("Using LSTM only")
    merged_features = conv
  elif no_lstm:
    print("Using global only")
    merged_features = input_global
  else:
    print("Using LSTM + global")
    merged_features = keras.layers.concatenate([input_global, conv])
  # fully-connected layers
  dropout_rate = 0.1
  dense = keras.layers.Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_1')(merged_features)
  dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_1')(dense)
  dense = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_2')(dense)
  dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_2')(dense)
  dense = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_3')(dense)
  dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_3')(dense)
  dense = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_4')(dense)
  dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_4')(dense)
  dense = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_5')(dense)
  dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_5')(dense)

  # output
  output = keras.layers.Dense(1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'output')(dense)
  optimizer = keras.optimizers.Adam()

  model = keras.models.Model(inputs = [input_global, input_objects], outputs = [output])
  model.compile(optimizer = optimizer, loss = 'binary_crossentropy')
  print((model.summary()))

  return model


def gru_v1(max_objects, n_features, n_global_features, no_global, no_lstm, n_lstm):
  input_objects = keras.layers.Input(shape=(max_objects, n_features), name = 'input_objects')
  input_global = keras.layers.Input(shape=(n_global_features,), name = 'input_global')

  maxnorm = 2
  # 1x1 convolutional layers
  dropout_rate = 0.15
  #conv = keras.layers.Convolution1D(64, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'conv_1')(input_objects)
  #conv = keras.layers.Dropout(rate = dropout_rate, name = 'conv_dropout_1')(conv)
  #conv = keras.layers.Convolution1D(64, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'conv_2')(conv)
  #conv = keras.layers.Dropout(rate = dropout_rate, name = 'conv_dropout_2')(conv)
  #conv = keras.layers.Convolution1D(32, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'conv_3')(conv)
  #conv = keras.layers.Dropout(rate = dropout_rate, name = 'conv_dropout_3')(conv)
  #conv = keras.layers.Convolution1D(6, 1, kernel_initializer = 'lecun_uniform', activation = 'relu', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'conv_4')(conv)
  #conv = keras.layers.Dropout(rate = dropout_rate, name = 'conv_dropout_4')(conv)

  conv = input_objects

  # lstm
  batch_momentum = 0.6
  dropout_rate = 0.1

  gru = keras.layers.GRU(n_lstm, implementation=1, name='gru_1', return_sequences=True)(conv)
  gru = keras.layers.Dropout(rate = dropout_rate, name = 'gru_dropout_1')(gru)
  gru = keras.layers.GRU(n_lstm, implementation=1, name='gru_2', return_sequences=False)(gru)
  gru = keras.layers.Dropout(rate = dropout_rate, name = 'gru_dropout_2')(gru)

  if no_global:
    print("Using LSTM only")
    merged_features = gru
  elif no_lstm:
    print("Using global only")
    merged_features = input_global
  else:
    print("Using LSTM + global")
    merged_features = keras.layers.concatenate([input_global, gru])

  # fully-connected layers
  dropout_rate = 0.2
  dense = keras.layers.Dense(300, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_1')(merged_features)
  dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_1')(dense)
  dense = keras.layers.Dense(300, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_2')(dense)
  dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_2')(dense)
  #dense = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_3')(dense)
  #dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_3')(dense)
  #dense = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_4')(dense)
  #dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_4')(dense)
  #dense = keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'dense_5')(dense)
  #dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout_5')(dense)

  # output
  output = keras.layers.Dense(1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', kernel_constraint = keras.constraints.max_norm(maxnorm), name = 'output')(dense)
  optimizer = keras.optimizers.Adam()

  model = keras.models.Model(inputs = [input_global, input_objects], outputs = [output])
  model.compile(optimizer = optimizer, loss = 'binary_crossentropy')
  print((model.summary()))

  return model

