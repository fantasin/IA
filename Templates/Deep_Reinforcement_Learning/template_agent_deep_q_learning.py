import random

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential



class Agent_DQN:
    def __init__(self,number_element_state, number_action, number_hidden_layer,neurones, gamma,alphan,epsilon,epsi_min,decrease_espilon,buffer_size):
        self.number_action =number_action
        self.buffer_size =buffer_size
        self.gamme,self.alpha,self.epsilon,self.decrease_espilon = gamma,alphan,epsilon,decrease_espilon
        self.main_network = self._create_network(number_element_state, number_action, number_hidden_layer,neurones)
        self.target_network = self._create_network(number_element_state, number_action, number_hidden_layer,neurones )
        self.epsi_min = epsi_min

        self.action_replay_buffer = Action_replay_buffer(buffer_size,number_element_state)

        self.copy_neuronal_network()

    def learn(self,batch_size):
        if self.action_replay_buffer.cmpt > batch_size:
            old_observations_samples,choices_samples,rewards_samples,obsersavtions_samples,ends_samples = self.action_replay_buffer.get_sample(batch_size)
            q = self.main_network.predict(old_observations_samples,verbose=0)


            tmp  = np.arange(batch_size).reshape((batch_size,1)) #chreate a tempory index to take the batch


            #q_hat = (1-self.alpha) * q[tmp,choices_samples] + self.alpha * (rewards_samples + self.gamme * self._maxQ(obsersavtions_samples,batch_size)*(1-ends_samples))


            q_hat = rewards_samples + self.gamme * self._maxQ(obsersavtions_samples,batch_size)*ends_samples

            q[tmp,choices_samples] = (1-self.alpha) * q[tmp,choices_samples] + self.alpha * q_hat
            target  = q
            self.main_network.fit(old_observations_samples,target,epochs = 1,batch_size=batch_size, verbose=0)


    def _maxQ(self,batch_new_state,batch_size):
        rep = self.target_network.predict(batch_new_state,verbose=0)

        return np.max(rep,axis=1).reshape((batch_size,1))

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


        self.epsilon = max(self.epsilon * self.decrease_espilon,self.epsi_min)

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
        model.compile(optimizer="adam", loss="huber_loss", metrics=["acc"])
        return model


class Action_replay_buffer:
    def __init__(self,size,size_state):
        self.cmpt = 0
        self.MAX_SIZE = size
        self.old_observations = np.zeros((size,size_state))
        self.choices = np.zeros((size,1),dtype="int8")
        self.rewards = np.zeros((size,1))
        self.obsersavtions = np.zeros((size,size_state))
        self.ends = np.zeros((size,1))
    def add_elem(self,old_obs, choice, reward, observation,end):
        self.old_observations[self.cmpt%self.MAX_SIZE] = old_obs
        self.choices[self.cmpt%self.MAX_SIZE] = choice
        self.rewards[self.cmpt%self.MAX_SIZE] = reward
        self.obsersavtions[self.cmpt%self.MAX_SIZE] = observation
        self.ends[self.cmpt%self.MAX_SIZE] = 1 - end
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







