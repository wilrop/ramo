import pandas as pd
import numpy as np

provide_comms = True
criterion = 'SER'
rand_prob = False
episodes = 10000


def normalise(min, max, arr):
    normalised = []
    for i in arr:
        new = (i - min)/(max - min)
        normalised.append(new)
    return normalised


# ['game1', 'game2', 'game3', 'game4', 'game5']
for game in ['game1', 'game2', 'game3', 'game4', 'game5']:
    if game == 'game1':
        min1 = 8
        max1 = 16
        min2 = 0
        max2= 4
    elif game == 'game2':
        min1 = 8
        max1 = 16
        min2 = 0
        max2 = 4
    elif game == 'game3':
        min1 = 8
        max1 = 16
        min2 = 0
        max2 = 4
    elif game == 'game4':
        min1 = 5
        max1 = 17
        min2 = 2
        max2 = 6
    else:
        min1 = 5
        max1 = 17
        min2 = 2
        max2 = 6

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

        normalised1 = normalise(min1, max1, payoffs1)
        normalised2 = normalise(min2, max2, payoffs2)

        mean1 = np.mean(payoffs1)
        mean2 = np.mean(payoffs2)
        nmean1 = np.mean(normalised1)
        nmean2 = np.mean(normalised2)
        print("Average payoff in " + game + " for agent 1: " + repr(mean1))
        print("Average payoff in " + game + " for agent 2: " + repr(mean2))
        print("Average normalised payoff in " + game + " for agent 1: " + repr(nmean1))
        print("Average normalised payoff in " + game + " for agent 2: " + repr(nmean2))
