import numpy as np
import matplotlib.pyplot as plt


def reset_bandit(n_levers = 10, sigma=1, mu=1):
  """ create a bandit with a random normal distribution assigned to each level"""
  mus = np.random.normal(size=[n_levers])*sigma + mu
  vars = np.random.normal(size=[n_levers])*sigma + mu
  bandit = {'mus': mus, 'vars': vars, 'n_levers': n_levers}
  return bandit


def draw_lever( bandit, n ):
  """ generate a random value for lever n of the bandit """
  reward = np.random.normal()* bandit['vars'][n] + bandit['mus'][n]
  return reward


# Reset the bandit with appropriate parameters
n_levers = 20
bandit = reset_bandit(n_levers)
n_tries = 10000


# how to store a historical average of levers?
# storing all historical values is expensive
# instead we regress against past regressions of the lever values
# this accrues some error, so we update our values to mitigate error
# alpha = learning rate
# Error = Vn - Tn
# minimise the loss, i.e. the expected square error: Loss = E[(Vn - r)^2]
# Vn = current estimated value of reward for lever n
# r_n_i = reward for level n at iteration i
# with each sample, we move either "left" or "right" from the predicted expectation
# so we update the value by a small amount (not a large step otherwise this results in instability)

# Action selection logic
def action_selection(values, chosen, bandit, method='greedy'):
    if method == 'greedy':  # greedy algorithm
        return np.argmax(values)  # returns the index i.e. lever with the maximum reward in "values"

    if method == 'ucb':  # upper confidence bounds algorithm
        bonus = 4 * np.sqrt(np.log(i) / chosen)
        values[chosen] = values[chosen] + bonus
        return np.argmax(values)

    if method == 'eps':  # epsilon greedy algorithm
        # epsilon is an arbitrarily chosen value between 0 and 1,
        # representing the proprtion of times that we randomly choose a lever.
        # the rest of the time, we choose greedily (i.e. the lever with max reward)
        epsilon = 0.1
        # generate a bool with epilson probability of being True, and 1-epsilon prob of being False
        choose_greedily = (np.random.rand() >= epsilon)  # rand() returns a float [0,1)
        if choose_greedily:
            return np.argmax(values)
        else:  # else choose randomly
            return np.random.randint(n_levers)

    if method == 'opt':  # optimism in the face of uncertainty algorithm
        if np.random.uniform() < 0.05:
            return np.random.randint(bandit['n_levers'])
        else:
            return np.argmax(values)


# Learning code
alpha = 0.03  # learning rate, if 1 you would just choose the next best value
values = np.zeros(n_levers)  # [0,0,0,..,0] the stochastically generated reward value for the current iteration, i
chosen = np.ones(n_levers)  # [1,1,1,..,1] represents the number of times a lever (n = index) is chosen
# set to all 1s since for ubc, we divide by chosen[], so cannot be 0
rewards = np.zeros(n_tries)  # [0,0,0,..,0] the estimated reward we are regressively predicted
all_values = np.zeros([n_tries, n_levers])

for i in range(1, n_tries):  # 1 to 9999

    action = action_selection(values, chosen, bandit, 'eps')  # choose a lever based on a given method
    chosen[action] += 1  # increment the counter for the lever that was chosen
    reward = draw_lever(bandit, action)  # each time the lever is pulled, the reward is updated based on current value
    rewards[i] = reward  # set a new expected max reward, which was calculated from the lever that we chose

    # YOUR CODE
    error = reward - values[action]
    values[action] += alpha * error
    all_values[i] = values


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)

#Plotting Logic
window = n_tries //30
plt.plot(rewards, alpha = 0.2)
plt.plot(rewards * 0 + np.max(bandit['mus']), '--')
smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
plt.plot(smoothed,'r')

plt.subplot(1,2,2)
plt.plot(all_values)
plt.show()


# blue = historical returns
# red = predicted expected maximum reward
# orange = true maximum reward

# red line needs to converge on the orange line
# by default the code won't work

# next task - create a plot that shows how each of the n estimated values, Vn (y axis), is changing with each iteration, i (x axis)
# in the plot we should see that one lever suddenly shoots up at around 4000 iterations
# the reason because it overtakes the lever that had the highest estimated reward value
# that initial highest lever had a lot of samples taken, since it was the highest, so we had a strong estimate for what it was
# so we start exploring the new highest lever, which is under explored,
# but we suspect might be better because of it has slightly overtaken the first level

# problems with epsilon greedy:
  # it continues exploring each bad levers at around 5% frequency of iterations,
  # even when the estimated expected reward is super low

# next task - explore optimistic greedy algorithm rather than epsilon greedy
# optimism in the face of uncertainty = algorithm is more inclined to explore at the start due to optimistic initialisation,
# compared to the epsilon greedy algorithm

# next task - Upper Confidence Bounds (ucb)
# the problem with optimism in the face of uncertainty is we have to arbitrarily decide how optimistic we are at the start
# for ucb, we take Vn and
# we add a bonus or incentive to underexplored levers OR has the highest potential estimateed value
# (this potential comes from uncertainty about our estimate, which comes from not having chosen the lever enough to get an accurate estimate)
# so, underexploration and high potential may often be the same thing
# B[n] = Exploration bonus of lever n = 4 * sqrt(log(i)) / chosen[n]
# the 4 is an arbitrary (for now..) constant multiplier
