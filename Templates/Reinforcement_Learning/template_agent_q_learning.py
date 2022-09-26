import copy
import random

import numpy as np

class Agent_q_learning:
    def __init__(self,list_action,number_state,gamma,alpha,epsilon,decrease_espilon = None):
        """
        :param list_action: liste des différents actions (int)

        :param number_state: dictionnaire qui lie les états ensembles, la clef est l'état actuelle et la valeur
        est une liste d'état possiblement joignable

        :param gamma: discount value, entre 0 et 1

        :param alpha: learning rate, entre 0 et 1

        :param epsilon: for gready espilon, entre 0 et 1

        :param decrease_espilon: for espilon gready decrease, entre 0 et 1 si None désactive la fonctionalité

        """
        self.list_action = list_action
        self.number_state = number_state
        self.gamme = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.decrease_espilon = decrease_espilon
        self.number_action = len(list_action)
        self._create_q_table()


    def _create_q_table(self):
        self.q_table = np.zeros((self.number_state,self.number_action))

    def make_choice(self,current_state):
        random_value = random.random()
        if  random_value < self.epsilon:
            action = [random.choice(self.list_action)]
        else:
            best_action = None
            best_value = None
            for i in range(self.number_action):
                if best_action == None:
                    best_action = i
                    best_value = self.q_table[current_state,i]
                    action = [i]
                elif self.q_table[current_state,i] > best_value:
                    best_action = i
                    best_value = self.q_table[current_state,i]
                    action = [i]
                elif self.q_table[current_state,i] == best_value:
                    action.append(i)

        self.epsilon = self.epsilon * self.decrease_espilon
        rep = random.choice(action)

        return rep

    def learn(self,current_state,new_state,action,reward,done):


        if done :
            self.q_table[current_state,action] = (1-self.alpha) * self.q_table[current_state,action] \
                                              + self.alpha * (reward)
        else:
            self.q_table[current_state,action] = (1-self.alpha) * self.q_table[current_state,action] \
                                                  + self.alpha * (reward + self.gamme * self._maxQ(new_state))


    def _maxQ(self,new_state):
        best_q = float('-inf')
        for i in range(self.number_action):
            if self.q_table[new_state,i] > best_q:
                best_q = self.q_table[new_state,i]

        return best_q