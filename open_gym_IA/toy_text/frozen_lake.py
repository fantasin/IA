import gym
from Templates.Reinforcement_Learning.template_agent_q_learning import Agent_q_learning
from open_gym_IA.toy_text.template_game_toy_text import play_game

environement_train = gym.make('FrozenLake-v1',desc=None, map_name="4x4", is_slippery=True)
environement_test = gym.make('FrozenLake-v1',render_mode="human", desc=None, map_name="4x4", is_slippery=True)
ALPHA = 0.1
GAMMA = 0.999
EPSILON = 0.80
EPSILON_DECREASE = 0.9999

list_action = [*range(environement_train.action_space.n)]
number_state = [environement_train.observation_space.n]


agent = Agent_q_learning(list_action,number_state,GAMMA,ALPHA,EPSILON,EPSILON_DECREASE)

NUMBER_EPISODE = 10000
NUMBER_TEST = 1


play_game(environement_train,environement_test,agent,NUMBER_EPISODE,NUMBER_TEST)

