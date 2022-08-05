import numpy as np
import matplotlib.pyplot as plt


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
values = np.zeros([env['size'], env['size']])
q_values = np.zeros([env['size'], env['size'], 4])
chosen = np.ones([env['size'], env['size'], 4])

# Learning Logic
alpha = 0.5 # learning rate i.e. how much we rely on past actions to influence future actions

# run episode
rewards = np.zeros(n_episodes)
for e in range(n_episodes):
    s = reset(env)
    for t in range(max_length):
        action = action_selection(q_values, chosen, s, 'eps')
        chosen[s[0], s[1], action] += 1
        r, s_prime, done = step(env, action)
        rewards[e] += r  # log rewards

        # YOUR CODE HERE
        td_error = 0 # fix
        values[s[0], s[1]] += alpha * td_error
        td_error = 0  # fix
        q_values[s[0], s[1], action] += alpha * td_error
        s = s_prime
        if done:
            break


# Plotting logic
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(values)
plt.colorbar()

plt.subplot(1,2,2)
plt.plot(rewards, alpha = 0.2)
window = n_episodes // 30
smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
plt.plot(smoothed)
plt.show()
