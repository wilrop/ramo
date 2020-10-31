import pandas as pd
import numpy as np

provide_comms = False
criterion = 'SER'
rand_prob = False
episodes = 10000

# ['game1', 'game2', 'game3', 'game4', 'game5']
for game in ['game1', 'game2', 'game3', 'game4', 'game5']:
    for opt_init in [False]:  # [True, False]:
        path_data = f'data/{criterion}/{game}'

        if opt_init:
            path_data += '/opt_init'
        else:
            path_data += '/zero_init'

        if rand_prob:
            path_data += '/opt_rand'
        else:
            path_data += '/opt_eq'

        info = 'NE_'

        if provide_comms:
            info += 'comm'
        else:
            info += 'no_comm'

        df1 = pd.read_csv(f'{path_data}/agent1_{info}.csv')
        df2 = pd.read_csv(f'{path_data}/agent2_{info}.csv')
        payoffs1 = df1['Payoff']
        payoffs2 = df2['Payoff']
        mean1 = np.mean(payoffs1)
        mean2 = np.mean(payoffs2)
        print("Average payoff in " + game + " for agent 1: " + repr(mean1))
        print("Average payoff in " + game + " for agent 2: " + repr(mean2))
