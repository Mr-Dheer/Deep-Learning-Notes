import numpy as np
import itertools

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
      if s == self.final:
        if next_s == s:
          return 1
        else:
          return 0
      if n_s[a]==s_next:
          return 1
      else:
          return 0
      return 0

    # TODO insert code here

    # -------------------------

  def print_policy(self, policy):
    P = np.array(policy).reshape(self.side, self.side)
    print(self.actions_repr[P])
  
  def print_values(self, values):
    V = np.array(values).reshape(self.side, self.side)
    print(V)


def eval_policy(world, policy,values, gamma=0.9, theta=0.01):

    while True:
        vDel = 0
        for s in world.states:
            val = values[s]
            vsum = 0
            for next_s in world.states:
                vsum+=world.transition_prob(s, next_s, policy[s]) * (world.reward(s, next_s, policy[s]) + gamma * values[next_s]) 
            values[s]=vsum
            vDel = max(vDel, abs(val - values[s]))
#        return values
        if vDel < theta:
            return values

def improve_policy(world, policy, values, gamma=0.9):

    vStable = True
    for s in world.states:

        pol_old = policy[s]
        vlis = []
        for a in world.actions:
            vsum = 0
            for s_next in world.states:
                vsum += world.transition_prob(s, s_next, a) * (world.reward(s, s_next, a) + gamma * values[s_next])
            vlis.append(vsum)
        policy[s] = world.actions[np.argmax(vlis)]
        if pol_old != policy[s]:
            vStable = False
    return vStable


def policy_iteration(world, gamma=0.9, theta=0.01):
    policy = np.array([np.random.choice(world.actions) for s in world.states])
 #   policy = np.array([2 for s in world.states])
    print('Initial policy')
    world.print_policy(policy)
    # Initialize values to zero
    values = np.zeros_like(world.states, dtype=np.float32)
#    values = np.zeros(len(world.states))
    stable = False
    for i in itertools.count():
        print(f'Iteration {i}')
        values = eval_policy(world, policy, values, gamma, theta)
        world.print_values(values)
        stable = improve_policy(world, policy, values, gamma)
        world.print_policy(policy)
        if stable:
            break;

    return policy, values



def main():
    np.set_printoptions(precision=3, linewidth=180)

    problem = GridWorld(4)
#    val = np.zeros(len(problem.states))
#    pol = np.array([2 for i in problem.states])
#    val2 = eval_policy(problem,pol,val,gamma=.5)
    policy, values = policy_iteration(problem, 0.5, 0.01)

    problem.print_policy(policy)
    problem.print_values(values)
#    problem.print_values(val2)


if __name__=="__main__":
    main()
