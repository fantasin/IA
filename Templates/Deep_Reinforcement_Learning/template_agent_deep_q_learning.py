from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from copy import deepcopy
class Agent_DQN:
    def __init__(self,number_action,number_element_state,gamma,alpha,epsilon,decrease_espilon,
                 optimizer ,function_loss,number_hidden_layer,metric):
        self.gamme = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.decrease_espilon = decrease_espilon

        self.main_network = self._create_network(number_element_state,number_action,optimizer,function_loss,number_hidden_layer,function_activation,metric )
        self.target_network = deepcopy(self.main_network)

    def learn(self,old,new):
        pass #TODO

    def _maxQ(self,new_state):
        rep = self.target_network.predict(new_state)
        return max(rep)
    def make_choice(self,obs):
        prediction = self.main_network.predict(obs)


    def _do_action_replay(self):
        pass

    def _create_network(self,number_element_state,number_action,optimizer,function_loss,number_hidden_layer,function_activation,metric):
        model = Sequential()
        model.add(
            Dense(number_hidden_layer, activation=function_activation, input_shape=(number_element_state,)))
        for j in range(number_hidden_layer - 1):
            model.add(Dense(units=number_hidden_layer, activation=function_activation))


        model.add(Dense(number_action, activation=FUNCTION_OUTPOUT))
        model.compile(optimizer=optimizer, loss=function_loss, metrics=metric)
        return model







