from collections import Counter
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as pl
import cPickle as pickle
import gzip
import os

# os.environ["SDL_VIDEODRIVER"] = "dummy"

from SwingyMonkey import SwingyMonkey

# Pick a random best choice 
def draw_greedy(values):
    m = np.max(values)
    best = []
    for i, v in enumerate(values):
        if v == m:
            best.append(i)
    return npr.choice(best)

''' 

'''
class BoltzmannExplorer(object):
    def __init__(self, tau=100000., decay=0.99):
        self.tau = tau
        self.decay = decay

    def decide_action(self, values):
        if self.tau == 0:
            return draw_greedy(values)
        else:
            try:
                temperature = values / self.tau

                diff = 20 - np.max(temperature)
                if np.isinf(diff):
                    return draw_greedy(values)

                # make sure we keep the exponential bounded (between +20 and -20)
                temperature += diff
                if np.min(temperature) < -20:
                    for i, v in enumerate(temperature):
                        if v < -20:
                            temperature[i] = -20
                probabilities = np.exp(temperature)
                probabilities = np.divide(probabilities, np.sum(probabilities))

                s = np.sum(probabilities)
                if not s < 1.00001 or not s > 0.99999:
                    print(values, self.tau, temperature, probabilities, 1 - s)
                    raise ValueError()
                r = npr.random()
                s = 0
                for i, p in enumerate(probabilities):
                    s += p
                    if s > r:
                        return i
                return npr.randint(len(probabilities))
            finally:
                self.tau *= self.decay


class EpsilonGreedyExplorer(object):
    def __init__(self, epsilon=0.3, decay=0.999):
        self.epsilon = epsilon
        self.decay = decay

    def decide_action(self, values):
        try:
            if npr.random() < self.epsilon:
                return npr.randint(len(values))
            else:
                return draw_greedy(values)
        finally:
            self.epsilon *= self.decay


class BaseLearner(object):
    def __init__(self, explorer=EpsilonGreedyExplorer(), name = ''):
        self.filename = self.__class__.__name__ + name + '.pkl.gz'
        self.explorer = explorer
        self.epoch = 0
        self.last_state = None
        self.last_score = None
        self.last_action = None
        self.last_reward = None
        self.scores = []

    def new_epoch(self):
        self.epoch += 1
        self.last_state = None
        self.last_score = None
        self.last_action = None
        self.last_reward = None

    def end_epoch(self):
        self.scores.append(self.last_score)

    def action_callback(self, raw_state):
        new_state = self.process_state(raw_state)
        new_action = self.decide_action(new_state)

        self.last_state = new_state
        self.last_score = raw_state['score']
        self.last_action = new_action

        return new_action

    def process_state(self, raw_state):
        def round_multiple(x, base):
            return int(base * round(float(x) / base))

        tree_state = raw_state['tree']
        monkey_state = raw_state['monkey']

        monkey_loc = (monkey_state['top'] + monkey_state['bot']) / 2

        # dist_from_center = monkey_loc - 200
        # if abs(dist_from_center) >= 98:
        #     dist_from_center = round_multiple(dist_from_center, 5)
        # else:
        #     dist_from_center = round_multiple(dist_from_center, 13)


        dist_from_gap = monkey_loc - (tree_state['top'] + tree_state['bot']) / 2
        '''
        if abs(dist_from_gap) <= 127:
            dist_from_gap = round_multiple(dist_from_gap, 5)
        elif dist_from_gap > 0:
            dist_from_gap = round_multiple(dist_from_gap - 132, 9) + 132
        else:
            dist_from_gap = round_multiple(dist_from_gap + 132, 9) - 132
        '''
        # print 'Initial Distance from Gap is %.2f' % dist_from_gap
        sign = 0 if dist_from_gap == 0 else (dist_from_gap) / abs(dist_from_gap)

        dist_from_gap = sign * round(np.sqrt(abs(dist_from_gap/2)))

        dist_from_tree = round_multiple(tree_state['dist'], 50)

        velocity = monkey_state['vel']
        velocity = round_multiple(velocity, 10)

        # print 'Final Distance from Gap is %.2f' % dist_from_gap
        return dist_from_gap, dist_from_tree, velocity

    def decide_action(self, new_state):
        return npr.rand() < 0.1

    def reward_callback(self, reward):

        if reward < 0:
            self.last_reward = -1000
        elif reward == 0: 
            self.last_reward = 0
        else:
            self.last_reward = 1

    def load(self):
        if os.path.isfile(self.filename):
            with gzip.open(self.filename, 'rb') as infile:
                return pickle.load(infile)

        return self

    def save(self):
        with gzip.open(self.filename, 'wb') as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)


class ModelLearner(BaseLearner):
    def __init__(self, name = ''):
        super(ModelLearner, self).__init__(name = name)
        self.state_action_count = {}
        self.state_action_reward = {}
        self.state_transition_count = {}

    def process_state(self, raw_state):
        new_state = super(ModelLearner, self).process_state(raw_state)

        if new_state not in self.state_action_count:
            self.state_action_count[new_state] = np.zeros(2)
            self.state_action_reward[new_state] = np.zeros(2)
            self.state_transition_count[new_state] = [Counter(), Counter()]

        if self.last_state is not None:
            self.state_action_count[self.last_state][self.last_action] += 1
            self.state_transition_count[self.last_state][self.last_action][new_state] += 1

        return new_state

    def reward_callback(self, reward):
        super(ModelLearner, self).reward_callback(reward)

        self.state_action_reward[self.last_state][self.last_action] += self.last_reward


class QLearner(BaseLearner):
    def __init__(self, name = ''):
        super(QLearner, self).__init__(name = name)
        self.learning_rate = 0.1
        self.discount_rate = 0.95
        self.Q = {}

    def process_state(self, raw_state):
        new_state = super(QLearner, self).process_state(raw_state)

        if new_state not in self.Q:
            self.Q[new_state] = np.zeros(2)

        if self.last_state is not None:
            q = self.Q[self.last_state][self.last_action]
            self.Q[self.last_state][self.last_action] =\
                q + self.learning_rate * (self.last_reward + self.discount_rate * np.max(self.Q[new_state]) - q)

        return new_state

    def decide_action(self, new_state):
        return self.explorer.decide_action(self.Q[new_state])


class SARSALearner(BaseLearner):
    def __init__(self):
        super(SARSALearner, self).__init__()
        self.learning_rate = 0.3
        self.discount_rate = 0.95
        self.Q = {}

    def process_state(self, raw_state):
        new_state = super(SARSALearner, self).process_state(raw_state)

        if new_state not in self.Q:
            self.Q[new_state] = np.zeros(2)

        return new_state

    def decide_action(self, new_state):
        action = self.explorer.decide_action(self.Q[new_state])

        if self.last_state is not None:
            q = self.Q[self.last_state][self.last_action]
            self.Q[self.last_state][self.last_action] =\
                q + self.learning_rate * (self.last_reward + self.discount_rate * self.Q[new_state][action] - q)

        return action


class TDLearner(ModelLearner):
    def __init__(self, name = ''):
        super(TDLearner, self).__init__(name = name)
        self.learning_rate = 0.3
        self.discount_rate = 0.95
        self.V = Counter()

    def process_state(self, raw_state):
        new_state = super(TDLearner, self).process_state(raw_state)

        if self.last_state is not None:
            v = self.V[self.last_state]
            self.V[self.last_state] =\
                v + self.learning_rate * (self.last_reward + self.discount_rate * self.V[new_state] - v)

        return new_state

    def decide_action(self, new_state):
        values = np.zeros(2)

        state_action_rewards = self.state_action_reward[new_state]
        state_transition_counts = self.state_transition_count[new_state]
        for i, state_action_count in enumerate(self.state_action_count[new_state]):
            if state_action_count != 0:
                value = state_action_rewards[i] / state_action_count
                for state, count in state_transition_counts[i].iteritems():
                    value += count / state_action_count * self.V[state]
                values[i] = value

        return self.explorer.decide_action(values)

print "Enter description and iterations " 
description, iterations = raw_input().split()

learner = QLearner(name = description).load()

saved = True
for ii in xrange(int(iterations)):
    # Reset the state of the learner.
    learner.new_epoch()

    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,                      # Don't play sounds.
                         text="Epoch %d" % learner.epoch,  # Display the epoch on screen.
                         tick_length=0,                    # Make game ticks super fast.
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)

    # Loop until you hit something.
    while swing.game_loop():
        pass

    print "Epoch %d: %d" % (learner.epoch, learner.last_score)

    # Save score
    learner.end_epoch()
    saved = False

    if learner.epoch % 500 == 0:
        learner.save()
        saved = True

if not saved:
    learner.save()

indices = np.arange(1, len(learner.scores) + 1)
moving_average = np.convolve(learner.scores, np.repeat(1.0, 100) / 100, 'valid')

pl.plot(indices, learner.scores, '-')
pl.plot(indices[99:], moving_average, 'r--')
pl.title(learner.__class__.__name__ + " Scores with Epsilon Greedy Explorer")
pl.yscale('symlog', linthreshy=1)
pl.ylabel("Score")
pl.xlabel("Iteration")
pl.show()