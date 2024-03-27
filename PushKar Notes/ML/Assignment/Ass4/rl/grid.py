import numpy as np
from itertools import product
import pdb
#from learning.model_based import Problem
#from learning.model_based import policy_iteration
#from learning.model_based import value_iteration

class Problem:

    def __init__(self):
        self.states = []

    def p(self, s, next_s, action):
        return 0

    def r(self, s, next_s, action):
        return 0

    def a(self, s):
        return []
    
class GridProblem(Problem):

    def __init__(self, side=4, final=0):
        self.side = side
        self.states = range(side * side)

        self.final = np.array([0,15])

        self.actions = range(4)
        self.actions_repr = np.array(['r', 'l', 'u', 'd'])

        self.probs = np.zeros(
            (len(self.states), len(self.states), len(self.actions)))
        self.rewards = np.zeros(
            (len(self.states), len(self.states), len(self.actions)))

        for (s, next_s, action) in product(self.states, self.states, self.actions):
            self.probs[(s, next_s, action)] = self._p(s, next_s, action)
            self.rewards[(s, next_s, action)] = self._r(s, next_s, action)

    def rlud(self, s):
        i, j = s / self.side, s % self.side

        states = [s] * 4
        if j + 1 < self.side:
            states[0] = s + 1
        if j - 1 >= 0:
            states[1] = s - 1
        if i - 1 >= 0:
            states[2] = s - self.side
        if i + 1 < self.side:
            states[3] = s + self.side

        return states

    def p(self, s, next_s, action):
        return self.probs[s, next_s, action]

    def r(self, s, next_s, action):
        return self.rewards[s, next_s, action]

    def _p(self, s, next_s, action):
        if any(s == self.final):
            if next_s == s:
                return 1
            else:
                return 0
        elif self.rlud(s)[action] == next_s:
            return 1

        return 0

    def _r(self, s, next_s, action):
        if any(s == self.final):
            return 0

        return -1

    def a(self, s):
        return self.actions

    def print_policy(self, policy):
        P = np.array(policy).reshape(self.side, self.side)
        print(self.actions_repr[P])

    def print_values(self, values):
        V = np.array(values).reshape(self.side, self.side)
        print(V)

def policy_iteration(problem, gamma=0.9, theta=0.01):
    policy = np.array([np.random.choice(problem.a(s)) for s in problem.states])
    print("Initial Random policy is")
    problem.print_policy(policy)
    stable = False
    while not stable:
        values = eval_policy(problem, policy, gamma, theta)
        stable = improve_policy(problem, policy, values, gamma)
    return policy, values

def eval_policy(problem, policy, gamma=0.9, theta=0.01):
    value = np.zeros(len(problem.states))

    p = problem.p
    r = problem.r
    j=0
    while True:
        delta = 0
        for s in problem.states:
            v = value[s]
            #pdb.set_trace()
            value[s] = sum([p(s, next_s, policy[s]) * (r(s, next_s, policy[s]) + gamma * value[next_s]) for next_s in problem.states])
            delta = max(delta, abs(v - value[s]))
        if delta < theta:
            return value

def improve_policy(problem, policy, value, gamma=0.9):
    p = problem.p
    r = problem.r

    stable = True
    for s in problem.states:
        actions = problem.a(s)

        b = policy[s]
        vlis = [sum([p(s, next_s, a) * (r(s, next_s, a) + gamma * value[next_s]) for next_s in problem.states]) for a in actions]
        policy[s] = actions[np.argmax(vlis)]
        if b != policy[s]:
            stable = False
#    return policy
    return stable


def main():
    np.set_printoptions(precision=3, linewidth=180)

    problem = GridProblem(4)
#    pdb.set_trace()
#    print(problem.probs)
#    print(problem.rewards)
#    vpolicy= np.array([1 for i in problem.states])
#    problem.print_policy(vpolicy)
#    print(policy)

 #   val = eval_policy(problem,vpolicy,gamma=.5)
#    problem.print_policy(policy)
 #   problem.print_values(val)
 #   vPol = improve_policy(problem,vpolicy,val)
 #   problem.print_policy(vPol)
 #   policy, values = policy_iteration(problem, 0.9, 0.001)
 #   val = eval_policy(problem,policy)
    policy, values = policy_iteration(problem, 0.5, 0.01)

    #    policy, values = value_iteration(problem, 0.9, 0.001)
#    print("Final Policy is:")
    problem.print_policy(policy)
    problem.print_values(values)
#    print(problem.p(0,0,2))


if __name__ == "__main__":
    main()
