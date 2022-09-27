import gym
import time
from Templates.Reinforcement_Learning.template_agent_q_learning import Agent_q_learning
from open_gym_IA.toy_text.template_game_toy_text import play_game

environement_train = gym.make("Blackjack-v1")
environement_test = gym.make("Blackjack-v1",render_mode = "human")




ALPHA = 0.1
GAMMA = 0.999
EPSILON = 0.80
EPSILON_DECREASE = 0.9999

list_action = [0,1]
state_shape = (32,11,3)


agent = Agent_q_learning(list_action,state_shape,GAMMA,ALPHA,EPSILON,EPSILON_DECREASE)

NUMBER_EPISODE = 100000
NUMBER_TEST = 10


play_game(environement_train,environement_test,agent,NUMBER_EPISODE,NUMBER_TEST)
