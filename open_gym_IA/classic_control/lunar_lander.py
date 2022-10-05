from Templates.Deep_Reinforcement_Learning.template_agent_deep_q_learning import Agent_DQN

import gym
environement_training = gym.make(
    "LunarLander-v2")

environement_test = gym.make(
    "LunarLander-v2",
    render_mode="human")


ALPHA = 0.1
GAMMA = 0.999
EPSILON = 0.80
EPSILON_DECREASE = 0.9999999
EPSILON_MINIMAL = 0.1


number_element_state, number_action = 8,4

TIME_STEP_BEFORE_TRAINING = 20
TIME_STEP_BEFORE_COPY = 50
NUMBER_EPISODE = 1000
NUMBER_TEST = 1
BATCH_SIZE = 64

BUFFER_SIZE = 2000
number_hidden_layer,neurones = 2,256

agent = Agent_DQN(number_element_state, number_action, number_hidden_layer,neurones, GAMMA,ALPHA,EPSILON,EPSILON_MINIMAL,EPSILON_DECREASE,BUFFER_SIZE)

SHOW_EACH = 50


time_step = 1
print("Begin training")

environement = environement_training
for episode in range(NUMBER_EPISODE):
    finish = False
    truncated = False
    cumulative_reward = 0
    observation = environement.reset()[0]

    if (episode+1)%SHOW_EACH == 0:
        environement = environement_test

    else:
        environement = environement_training

    observation = environement.reset()[0]


    while not (finish or truncated):
        choice = agent.make_choice(observation)
        old_obs = observation
        observation, reward, finish, truncated, info = environement.step(choice)
        cumulative_reward += reward
        end = False
        if finish or truncated:
            end = True
        agent.add_into_action_buffer(old_obs, choice, reward, observation,end)

        if time_step%TIME_STEP_BEFORE_TRAINING == 0:
            agent.learn(BATCH_SIZE)

        if time_step%TIME_STEP_BEFORE_COPY == 0:
            agent.copy_neuronal_network()

        time_step += 1


    print("Episode :", episode, "Reward :",cumulative_reward)





agent.save()

environement_training.close()

agent.epsilon = 0


print("Begin testing")
for i in range(NUMBER_TEST):
    finish = False
    observation = environement_test.reset()[0]

    while not finish:
        choice = agent.make_choice(observation)
        observation, reward, finish, truncated, info = environement_test.step(choice)


environement_test.close()