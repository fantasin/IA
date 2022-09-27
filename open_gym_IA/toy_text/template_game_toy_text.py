def play_game(environement_training,environement_test,agent,NUMBER_EPISODE,NUMBER_TEST):
    print("Begin training")
    for episode in range(NUMBER_EPISODE):
        finish = False
        observation = environement_training.reset()[0]

        if (episode % 100 == 0):
            print("pourcentage ", round(episode / NUMBER_EPISODE, 3),"%")

        while not finish:
            choice = agent.make_choice(observation)
            old_obs = observation
            observation, reward, finish, truncated, info = environement_training.step(choice)
            agent.learn(old_obs, observation, choice, reward, finish)


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
