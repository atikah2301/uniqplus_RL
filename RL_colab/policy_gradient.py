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
  # With probability 'noise' we randomly replace the action
  if np.random.uniform() < env['noise']:
    action = np.random.randint(0, 4)
  env['agent_position'] =  np.clip(env['agent_position'] + env['actions'][action], 0, env['size'] - 1)
  reward = -1
  done = False
  # Reward computation
  if np.array_equal(env['agent_position'], env['trap']): # if you hit the wall
    reward = -10
  if np.array_equal(env['agent_position'], env['goal']): # if you reach the goal
    reward = 10
    done = True
  return reward, env['agent_position'], done


env = create_env(noise = 0.0)
max_length = 100 # Length x height of grid, travelling over whole grid
gamma = 0.99 # Discount factor
n_episodes = 25


# Action selection logic
def action_selection(parameter_vector): # My policy
  params = parameter_vector[s[0], s[1]]
  # print(params)
  probs = np.exp(params)
  probs = probs / probs.sum() # Still a list (we applied softmax formula)
  action = np.random.choice([0,1,2,3], p=probs) # A weighted choice index 0 to 3
  return action


# Value logic
values = np.zeros([env['size'], env['size']]) # How much reward we anticipate for standing on each square
alpha = 0.03 # Learning rate

# run episode
rewards = np.zeros( n_episodes )
parameter_vector = np.ones([env['size'], env['size'], 4]) # One 'theta' for every coordinate (state) and action

for e in range( n_episodes ):
  s = reset(env)
  for t in range( max_length ):
    action = action_selection(parameter_vector) # Up, down, left or right. An index
    r, s_prime, done = step(env, action)
    rewards[e] += r # Total reward for each episode
    td_error = r + gamma * values[s_prime[0], s_prime[1]] * (not done) - values[s[0], s[1]] # Temporal difference error
    values[s[0], s[1]] += alpha * td_error
    # The derivative of softmax, should be same shape as parameter_vector
    softmax_derivative = parameter_vector * np.exp(parameter_vector)
    softmax_derivative /= np.sum(np.exp(parameter_vector) / np.sum(parameter_vector))
    print(f"softmax_derivative {softmax_derivative}, values {values[s[0],s[1]]}")
    parameter_vector += alpha * softmax_derivative * values[s[0], s[1]]
    #print(values, "\n")
    s = s_prime
    if done:
      break