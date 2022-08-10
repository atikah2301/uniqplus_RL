import numpy as np
import matplotlib.pyplot as plt
import seaborn


def create_env(size = 10, noise = 0): # int, int -> dict
  """ Create basic gridworld environment """
  agent_position = np.array([0,0]) # start at origin, (0,0)
  goal = np.array([size-1, size-1]) # set goal at (9,9), i.e. the opposite corner
  trap = np.array([6,6]) # set a "wall"
  # Action encoding: up / down / left / right as lists
  # To choose an action, choose its index 0,1,2,3
  actions =[[1,0], [-1,0], [0,-1], [0,1]]
  # store attributes of the environment in a dictionary to return
  env = {'agent_position':agent_position, 'goal':goal, 'trap': trap,
         'size':size, 'actions': actions, 'noise': noise }
  return env


def reset(env): # dict -> np.array[(int, int)]
  env['agent_position'] = np.array([0,0])
  return env['agent_position']


# Define environment dynamics
def step( env, action ): # dict, int -> int, np.array[(int, int)], bool

    # With probability 'noise' we randomly replace the chosen action with a random action
    if np.random.uniform() < env['noise']:
        action = np.random.randint(0, 4)

    # Carry out this action, assign it to the environment dictionary as current position + action = new position
    # use np.clip to ensure the new coordinate is at most 9 and at least 0 (i.e. don't leave the grid space!)
    env['agent_position'] =  np.clip(env['agent_position'] + env['actions'][action], 0, env['size'] - 1)
    reward = -1 # since a time step has passed, you are penalised for wasting time
    is_goal_found = False

    # Reward computation
    if np.array_equal(env['agent_position'], env['trap']): # if you hit the wall
        reward = -10
    if np.array_equal(env['agent_position'], env['goal']): # if you reach the goal
        reward = 10
        is_goal_found = True
    return reward, env['agent_position'], is_goal_found


# Create env
env = create_env(noise=0.0)  # noise = proportion of times the environment causes agent to behave randomly
max_length = 100  # path length capped since naive navigation gives a path length of 100 in the worst case
gamma = 0.99  # discount rate in bellman equation - how much we consider future rewards in calculating return
n_episodes = 200 # originally 5000


# Action selection logic
def action_selection(q_values, chosen, s, method='greedy'):
    """
    chosen = list where index i has the number of times lever i has been chosen for a trial in the current episode
    q values = action values i.e. all future reward associated with action a at state s
    values = state values i.e. all future reward associated with state s
    s = current state as a list of length 2, giving the current coordinate position of the agent in the form [x, y]
    """
    values = q_values[s[0], s[1]]

    if method == 'greedy':
        return np.argmax(values)

    if method == 'ucb':
        # YOUR CODE HERE
        return 0  # fix

    if method == 'eps':
        # YOUR CODE HERE
        return 0  # fix


# Value logic
values = np.zeros([env['size'], env['size']]) # 10x10 states values
q_values = np.zeros([env['size'], env['size'], 4]) # 10x10x4 q-table for all 100 states and 4 actions, initialised at 0
chosen = np.ones([env['size'], env['size'], 4]) # 10x10x4 counting the number of times action a is taken in state s

# Learning Logic
alpha = 0.5 # learning rate i.e. how much we rely on past actions to influence future actions

# run episode
rewards = np.zeros(n_episodes) # track the total reward for each episode
for e in range(n_episodes):
    s = reset(env) # reset the agent's position to [0, 0] for new episode
    for t in range(max_length): # 100 time steps
        action = action_selection(q_values, chosen, s, 'greedy') # choose an action {up, down, left, right}
        # given as a direction vector like [0, 1] for up
        chosen[s[0], s[1], action] += 1 # increment the chosen state-action's counter
        r, s_prime, done = step(env, action) # calculate reward, determine next state, check if goal is met
        rewards[e] += r  # add to the total reward for this episode

        # YOUR CODE HERE
        td_error = r + gamma * values[s_prime[0], s_prime[1]] * (not done) - values[s[0], s[1]]
        values[s[0], s[1]] += alpha * td_error # update state values
        td_error = r + gamma * q_values[s_prime[0], s_prime[1], action] * (not done) - q_values[s[0], s[1], action]
        q_values[s[0], s[1], action] += alpha * td_error # update action values
        s = s_prime # take the next state
        if done:
            break


# Plotting logic
plt.figure(figsize=(10,4)) # create figure, set width x height in inches
plt.subplot(1,2,1) # subplot(rows, cols, current index to plot on numbered starting with 1)
# i.e. create a plotting space with 1 row and 2 columns, and set the first plot at index 1
seaborn.heatmap(values, cmap="flare", linewidths=.5) # show the 2D array values as an image (heatmap by default)
# plt.imshow(values)
# plt.colorbar() # show the colour scale bar for image
plt.title("Gridworld State Values")

plt.subplot(1,2,2) # set the second plot at index 2 in the plotting space
plt.plot(rewards, alpha = 0.2) # plot the rewards for each episode (semi-transparent blue line)
window = n_episodes // 30
smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
plt.plot(smoothed) # plot a red line showing a smoother plot for the rewards
# this shows the trend more clearly
plt.title("Episode Rewards")
plt.show()

