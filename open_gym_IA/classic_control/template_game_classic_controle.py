import copy
import random


def play_game(environement_training,environement_test,agent,SIZE_REPLAY_EXPERIENCE,TIME_STEP_BEFORE_TRAINING,TIME_STEP_BEFORE_COPY,NUMBER_EPISODE,NUMBER_TEST):

    MEMORY_REPLAY = []

    time_step = 1
    print("Begin training")
    for episode in range(NUMBER_EPISODE):
        finish = False
        truncated = False

        observation = environement_training.reset()[0]

        if (episode % 100 == 0):
            print("pourcentage ", round(episode / NUMBER_EPISODE, 3),"%")

        while not (finish or truncated):
            choice = agent.make_choice(observation)
            old_obs = observation
            observation, reward, finish, truncated, info = environement_training.step(choice)

            if time_step%TIME_STEP_BEFORE_TRAINING == 0:
                tmp = copy.deepcopy(MEMORY_REPLAY)
                random.shuffle(tmp)
                for e in tmp:
                    old_obs, choice, reward, observation,end = e
                    agent.learn(old_obs, observation, choice, reward, end)

            end = False
            if finish or truncated:
                end = True

            e = (old_obs, choice, reward, observation,end)
            if len(MEMORY_REPLAY) == SIZE_REPLAY_EXPERIENCE:
                MEMORY_REPLAY.pop(0)
                MEMORY_REPLAY.append(e)
            else:
                MEMORY_REPLAY.append(e)

            if time_step%TIME_STEP_BEFORE_COPY == 0:
                agent.copy_neuronal_network()


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
