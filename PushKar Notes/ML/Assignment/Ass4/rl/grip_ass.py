import numpy as np
import itertools
import pdb
np.set_printoptions(precision=3, linewidth=180)

class GridWorld:
  UP = 0
  DOWN = 1
  LEFT = 2
  RIGHT = 3

  def __init__(self, side=4):
    self.side = side
    # -------------------------
    # Define integer states, actions, and final states as specified in the problem description

    # TODO insert code here
    self.actions = range(4)
    self.states = range(self.side*self.side)
    self.finals = np.array([0,15])

    # -------------------------
    self.actions_repr = np.array(['↑', '↓', '←', '→'])

  def reward(self, s, s_next, a):
    # -------------------------
    # Return the reward for the given transition as specified in the problem description
    if any(s==self.finals):
        return 0
    else:
        return -1
    # TODO insert code here

    # -------------------------

  def transition_prob(self, s, s_next, a):
      
      # -------------------------
      # Return a probability in [0, 1] for the given transition as specified in the problem description      
      # neighbours:
      # column and row number of current state
      r,c = s//self.side,s%self.side
      n_s = [s]*4
      
      if r-1>=0:
          n_s[0] = s-self.side 
      if r+1<self.side:
          n_s[1] = s+self.side
      if c-1>=0:
          n_s[2] = s-1
      if c+1<self.side:
          n_s[3] = s+1
      n_s = np.array(n_s)
      # TODO insert code here
      if any(s==self.finals):
          if s_next==s:
              return 1
          else:
              return 0
 #     pdb.set_trace()
      if n_s[a]==s_next:
          return 1
      else:
          return 0
      return 0
      # -------------------------

  def print_policy(self, policy):
    P = np.array(policy).reshape(self.side, self.side)
    print(self.actions_repr[P])
  
  def print_values(self, values):
    V = np.array(values).reshape(self.side, self.side)
    print(V)




def eval_policy(world, policy, values,gamma=0.9, theta=0.01):
  while True:
    vDel = 0
    for i in world.states:
      #print("State is {}".format(i))
      v = values[i]
      vsum = 0
      for s_n in world.states:
        vsum += world.transition_prob(i, s_n, policy[i]) * (world.reward(i, s_n, policy[i]) + gamma * values[s_n])
        values[i] = vsum    
      vDel = max(vDel, abs(v - values[i]))

    if vDel < theta:
      return values

def improve_policy(world, policy, values, gamma=0.9):
  vStable = False
  while not vStable:
    for s in world.states:
      p = policy[s]
      vA = []
      for a in world.actions:
        vsum = 0
        for n_s in world.states:
          vsum+= world.transition_prob(s,n_s,a)*(world.reward(s,n_s,a)+gamma*values[n_s])
        vA.append(vsum)
      policy[s] = np.argmax(vA)
      if p != policy[s]:
        vStable=True

    return vStable


def policy_iteration(world, gamma=0.9, theta=0.01):
  # Initialize a random policy
  policy = np.array([np.random.choice(world.actions) for s in world.states])
  print('Initial policy')
  world.print_policy(policy)
  # Initialize values to zero
  values = np.zeros_like(world.states, dtype=np.float32)

  # Run policy iteration
  stable = False
  for i in itertools.count():
    print(f'Iteration {i}')
    values = eval_policy(world, policy, values, gamma, theta)
    world.print_values(values)
    stable = improve_policy(world, policy, values, gamma)
    world.print_policy(policy)
    if stable:
      break

  return policy, values

world = GridWorld()
vPol = np.array([2 for i in world.states])
value = np.zeros(len(world.states))

#val = eval_policy(world,vPol,value,gamma=.5)
#world.print_values(val)
#val2 = eval_policy(world,vPol,val,gamma=.5)
#world.print_values(val2)

policy, values = policy_iteration(world, gamma=0.5)
world.print_policy(policy)
world.print_values(values)
pdb.set_trace()
