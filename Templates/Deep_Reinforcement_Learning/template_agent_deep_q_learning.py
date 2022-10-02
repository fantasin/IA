import random

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential



class Agent_DQN:
    def __init__(self,number_element_state, number_action, number_hidden_layer,neurones, gamma,alphan,epsilon,decrease_espilon,buffer_size):
        self.number_action =number_action
        self.buffer_size =buffer_size
        self.gamme,self.alpha,self.epsilon,self.decrease_espilon = gamma,alphan,epsilon,decrease_espilon
        self.main_network = self._create_network(number_element_state, number_action, number_hidden_layer,neurones)
        self.target_network = self._create_network(number_element_state, number_action, number_hidden_layer,neurones )

        self.action_replay_buffer = Action_replay_buffer(buffer_size)

        self.copy_neuronal_network()

    def learn(self,batch_size):
        if self.action_replay_buffer.cmpt > batch_size:
            old_observations_samples,choices_samples,rewards_samples,obsersavtions_samples,ends_samples = self.action_replay_buffer.get_sample(batch_size)

            list_action = self.main_network.predict(old_observations_samples,verbose=0)
            print(list_action)

            """q = list_action[0][action]
            q = (1-self.alpha) * q + self.alpha * (reward + self.gamme * self._maxQ(new_state)*(1-done))
            target = list_action
            target[0][action] = q
            self.main_network.fit(current_state,target,epochs = 1, verbose=0)"""


    def _maxQ(self,new_state):
        rep = self.target_network.predict(new_state,verbose=0)
        return max(rep[0])

    def add_into_action_buffer(self,current_state,new_state,action,reward,done):
        self.action_replay_buffer.add_elem(current_state,new_state,action,reward,done)

    def make_choice(self,obs):
        random_value = random.random()
        if random_value < self.epsilon:
            action = random.choice([*range(self.number_action)])
        else:
            obs = np.array([obs])
            prediction = self.main_network.predict(obs,verbose=0)
            action = np.argmax(prediction)


        self.epsilon = self.epsilon * self.decrease_espilon

        return action


    def copy_neuronal_network(self):
        self.target_network.set_weights(self.main_network.get_weights())


    def _create_network(self, number_element_state, number_action, number_hidden_layer,neurones):
        model = Sequential()
        model.add(
            Dense(number_hidden_layer, activation="relu", input_shape=(number_element_state,)))
        for j in range(number_hidden_layer - 1):
            model.add(Dense(units=neurones, activation="relu"))
        model.add(Dense(number_action, activation="linear"))
        model.compile(optimizer="adam", loss="mse", metrics=["acc"])
        return model


class Action_replay_buffer:
    def __init__(self,size):
        self.cmpt = 0
        self.MAX_SIZE = size
        self.old_observations = np.zeros((1,size))
        self.choices = np.zeros((1,size))
        self.rewards = np.zeros((1,size))
        self.obsersavtions = np.zeros((1,size))
        self.ends = np.zeros((1,size))
    def add_elem(self,old_obs, choice, reward, observation,end):
        self.old_observations[self.cmpt%self.MAX_SIZE] = old_obs
        self.choices[self.cmpt%self.MAX_SIZE] = choice
        self.rewards[self.cmpt%self.MAX_SIZE] = reward
        self.obsersavtions[self.cmpt%self.MAX_SIZE] = observation
        self.ends[self.cmpt%self.MAX_SIZE] = end
        self.cmpt +=1

    def get_sample(self,batch_size):
        """
        Create a sample for the batch size
        :param batch_size: number element to take
        :return: tuple e

        Warning : a verification before need to be to just to check if we have enough element for the batch size
        """
        maximal_index = min(self.MAX_SIZE,self.cmpt)
        e = None
        if batch_size < maximal_index:
            batch = np.random.choice(maximal_index,batch_size)
            old_observations_samples = self.old_observations[batch]
            choices_samples = self.choices[batch]
            rewards_samples = self.rewards[batch]
            obsersavtions_samples = self.obsersavtions[batch]
            ends_samples = self.ends[batch]
            e = (old_observations_samples,choices_samples,rewards_samples,obsersavtions_samples,ends_samples)
        return e







