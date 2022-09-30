from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from copy import deepcopy

import tensorflow as tf
from tensorflow import keras

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
loss_function = keras.losses.Huber()


class Agent_DQN:
    def __init__(self,environement_information, IA_information, parameter_information):

        self.gamme,self.alpha,self.epsilon,self.decrease_espilon = parameter_information
        self.main_network = self._create_network(environement_information, IA_information )
        self.target_network = deepcopy(self.main_network)

    def learn(self,current_state,new_state,action,reward,done):
        list_action = self.main_network.predict(current_state)
        q = list_action[action]
        old_q = q
        if done :
            q = (1-self.alpha) * q + self.alpha * (reward)
        else:
            q = (1-self.alpha) * q + self.alpha * (reward + self.gamme * self._maxQ(new_state))


        loss = loss_function(q, old_q)

        # Backpropagation
        grads = tf.GradientTape().gradient(loss, self.main_network.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.main_network.trainable_variables))



    def _maxQ(self,new_state):
        rep = self.target_network.predict(new_state)
        return max(rep)

    def make_choice(self,obs):
        prediction = self.main_network.predict(obs)
        max_value = max(prediction)
        max_index = prediction.index(max_value)
        return max_index


    def copy_neuronal_network(self):
        self.target_network = deepcopy(self.main_network)


    def _create_network(self, environement_information, IA_information):
        number_element_state, number_action = environement_information
        optimizer, function_loss, number_hidden_layer, function_activation, metric = IA_information

        model = Sequential()
        model.add(
            Dense(number_hidden_layer, activation=function_activation, input_shape=(number_element_state,)))
        for j in range(number_hidden_layer - 1):
            model.add(Dense(units=number_hidden_layer, activation=function_activation))



        model.add(Dense(number_action, activation="linear"))
        model.compile(optimizer=optimizer, loss=function_loss, metrics=metric)
        return model







