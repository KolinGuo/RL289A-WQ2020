'''
## Network ##
# Defines the DQN network - architecture, training step and testing step. 
@author: Kolin Guo
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
import logging

class DQNModel:
    def __init__(self, state_shape, num_actions, learning_rate=0.00025, load_model_path=None, name='DQN'):
        # Get logger for network
        self.logger = logging.getLogger(name)

        self.state_shape = state_shape
        self.num_actions = num_actions

        # Create DQN model
        # Set backend float dtype
        keras.backend.set_floatx('float32')
        # Input dim: (batch, H, W, D, channels) = (32, 10, 10, 7, 4)
        inputs = keras.Input(shape=self.state_shape, dtype='float32', name='state')
        x = layers.Conv3D(32, (7, 7, 5), 1, padding='same', activation='relu', name='conv1')(inputs)
        x = layers.Conv3D(64, (5, 5, 5), 1, padding='same', activation='relu', name='conv2')(x)
        x = layers.Conv3D(64, (3, 3, 3), 1, padding='same', activation='relu', name='conv3')(x)
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(512, activation='relu', name='d1')(x)
        outputs = layers.Dense(self.num_actions, name='d2')(x)
    
        self.model = keras.Model(inputs, outputs, name=name)

        # Create optimizer
        self.optimizer = optimizers.RMSprop(learning_rate, momentum=0.95, epsilon=0.01)

        # Create loss function
        #self.loss_func = losses.MeanSquaredError()
        self.loss_func = losses.Huber()  # less sensitive to outliers (linearized MSE when |x| > delta)
        # Accumulate training loss
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.train_loss.reset_states()

        # Create checkpoint
        self.checkpoint = tf.train.Checkpoint(model=self.model, 
                                              optimizer=self.optimizer)
        if load_model_path is None: 
            self.logger.info('__init__: Creating a new DQN model')
        else:    # Restart training from the checkpoint
            self.checkpoint.restore(load_model_path)
            self.logger.info('__init__: Loading an existing DQN model from "%s"', load_model_path)

    def print_model_summary(self):
        self.model.summary()

    def plot_model(self, show_shapes=True):
        keras.utils.plot_model(self.model, show_shapes=show_shapes)

    def save_model(self, save_path, ckpt_number=None):
        # Save (only model weights) for update
        if ckpt_number is None: 
            self.checkpoint = tf.train.Checkpoint(model=self.model)
            ckpt_manager = tf.train.CheckpointManager(self.checkpoint, 
                    directory=save_path, max_to_keep=1, checkpoint_name='ckpt')
        else:   # Save (both model weights and optimizer) during training
            self.checkpoint = tf.train.Checkpoint(model=self.model, 
                                                  optimizer=self.optimizer)
            ckpt_manager = tf.train.CheckpointManager(self.checkpoint, 
                    directory=save_path, max_to_keep=None, checkpoint_name='ckpt')

        ckpt_path = ckpt_manager.save(ckpt_number)

        self.logger.info('Saving the model to %s', ckpt_path)

    # Only called for updating DQN_target
    def load_model(self, load_path):
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        ckpt_manager = tf.train.CheckpointManager(self.checkpoint, 
                    directory=load_path, max_to_keep=1, checkpoint_name='ckpt')

        self.checkpoint.restore(ckpt_manager.latest_checkpoint).assert_consumed()

        self.logger.info('Loading the model from %s', ckpt_manager.latest_checkpoint)

    # Train a step with a batch of states
    @tf.function
    def train_step(self, states, actions, targetQs):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            Q_vals = self.model(states, training=True)
            actions_one_hot = tf.one_hot(actions, self.num_actions, on_value=1.0, off_value=0.0, dtype=tf.float32)
            Q_vals_actions = tf.math.multiply(Q_vals, actions_one_hot, name='element-wise_Multiply')
            Q_vals_actions = tf.math.reduce_sum(Q_vals_actions, axis=1, name='reduce_sum')
            loss = self.loss_func(targetQs, Q_vals_actions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables), name='apply_gradient')

        # Accumulate training loss
        self.train_loss.update_state(loss)
    
    # Get accumulated training loss for an epoch and reset
    def get_training_loss(self):
        avg_loss = self.train_loss.result().numpy()
        self.train_loss.reset_states()
        return avg_loss

    # Predict an action given a state
    @tf.function
    def predict(self, state):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        Q_vals = self.model(state, training=False)
        return tf.math.argmax(Q_vals, axis=1)

    # Infer the network for Q(S, A) given a state
    @tf.function
    def infer(self, state):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        Q_vals = self.model(state, training=False)
        return Q_vals

