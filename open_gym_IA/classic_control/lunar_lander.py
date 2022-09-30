from template_game_classic_controle import play_game
from Templates.Deep_Reinforcement_Learning.template_agent_deep_q_learning import Agent_DQN

import gym
environement_training = gym.make(
    "LunarLander-v2",
    continuous = False,
    gravity = -10.0,
    enable_wind = False,
    wind_power = 15.0,
    turbulence_power = 1.5)

environement_test = gym.make(
    "LunarLander-v2",
    continuous = False,
    gravity = -10.0,
    enable_wind = False,
    wind_power = 15.0,
    turbulence_power = 1.5,
    render_mode="human")


ALPHA = 0.1
GAMMA = 0.999
EPSILON = 0.80
EPSILON_DECREASE = 0.9999


parameter_information =  (GAMMA, ALPHA, EPSILON, EPSILON_DECREASE)
number_element_state, number_action = environement_information
optimizer, function_loss, number_hidden_layer, function_activation, metric = IA_information

environement_information = ()

agent = Agent_DQN()

play_game()