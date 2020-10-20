import os
import errno


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def factorial(number):
    if number == 0 or number == 1:
        return 1
    else:
        return number * factorial(number - 1)


# calculate n C k (i.e. the number of combinations)
def choose(n, k):
    result = int(factorial(n)) / int(int((factorial(k)) * int(factorial(n - k))))
    return int(result)


# calculate the number of ways to place n indistinguishable items into k labelled (distinct) containers
def stars_and_bars(n, k):
    return choose(int(n + k - 1), int(n))


# generates the list of probability distributions which will be used when searching the strategy space
# n is the number of divisions (e.g. if n=100, the step size when generating all probability distributions is 0.01)
# k is the number of actions
# the probability vectors generated are of length k-1, as the probability of the last action may be calculated
def generate_stars_and_bars_lookup(n, k):
    step_size = 1.0 / n
    dists = generate_stars_and_bars(n, k)
    lookup = []
    for d in dists:
        candidate = [0] * (k - 1)
        for a in range(len(candidate)):
            candidate[a] = d[a] * step_size
        lookup.append(candidate)
    return lookup


# generate all possible ways to place n indistinguishable items into k labelled (distinct) containers
def generate_stars_and_bars(n, k):
    dists = []
    possible_dist = [0] * k
    possible_dist[0] = n
    dists.append(possible_dist.copy())
    left_bar = 0
    right_bar = 1
    while possible_dist[left_bar] > 0:
        possible_dist[left_bar] -= 1
        possible_dist[right_bar] += 1
        dists.append(possible_dist.copy())
        if right_bar != k - 1:
            generate_dists_recursive(dists, possible_dist, k, left_bar + 1, right_bar + 1)
    return dists


def generate_dists_recursive(dists, possible_dist, bars, left_bar, right_bar):
    while possible_dist[left_bar] > 0:
        possible_dist[left_bar] -= 1
        possible_dist[right_bar] += 1
        dists.append(possible_dist.copy())
        if right_bar != bars - 1:
            generate_dists_recursive(dists, possible_dist, bars, left_bar+1, right_bar+1)
    possible_dist[left_bar] += possible_dist[right_bar]
    possible_dist[right_bar] = 0



