import numpy as np
import pandas as pd
import random
import math

UNIT = 40
MAZE_H = 6
MAZE_W = 6

class Agent:
    ### START CODE HERE ###

    def __init__(self, actions):
        self.actions = actions
        self.epsilon = 0.01

    def choose_action(self, observation):
        action = np.random.choice(self.actions)
        return action

class DQN_Agent():
    def __init__(self, actions):
        self.actions = actions
        self.epsilon = 0
        self.q_table = {}
        self.alpha = 0.4
        self.gamma = 0.8
        self.model = {}
        self.counter = {}  # counter  for state
        self.N_counter = {}  # counter for state-action
        self.lamb_da = 0.95
        self.dyna_N = 10
        self.k = 5

        for f in [False, True]:
            for i in range(MAZE_H):
                for j in range(MAZE_W):
                    state = (float(j*UNIT+5), float(i*UNIT+5),
                                  float((j+1)*UNIT-5), float((i+1)*UNIT-5), f)
                    self.q_table[state] = [0,0,0,0]  # init q table
                    self.counter[state] = 1
                    self.N_counter[state] = np.array([1,1,1,1])

    def load_q_table(self,q_table):
        self.q_table = q_table


    def choose_action(self, s):
        #print(self.q_table)
        if len(s)==4:
            s.append(False)
        state = tuple(s)
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            c= []
            for i in range(4):
                if self.q_table[state][i] == max(self.q_table[state]):
                    c.append(i)
            action = int(random.choice(c))
        # #print(action)
        if (action == 0 and state[1] == 5.0) or\
            (action == 1 and state[3] == 235.0) or\
            (action == 2 and state[2] == 235.0) or\
            (action == 3 and state[0] == 5.0):
            self.q_table[state][action] = -1000000.0
            return self.choose_action(list(s))
        else:
            return action

    def update_q(self, s,s_,action,reward):
        state = tuple(s)
        state_ = tuple(s_)

        self.counter[state] += 1
        self.N_counter[state][action] += 1
        #print(state)

        self.q_table[state][action] += self.alpha*(reward
            +self.gamma*max([(self.q_table[state_][i]-self.N_counter[state_][i]/self.k) for i in range(4)])
            -self.q_table[state][action])

        # self.q_table[state][action] += self.alpha*(reward
        #     +self.gamma*max([(self.q_table[state_][i]+1/self.N_counter[state_][i]) for i in range(4)])
        #     -self.q_table[state][action])

        # self.q_table[state][action] += self.alpha*(reward
        #     +self.gamma*max(self.q_table[state_])
        #     -self.q_table[state][action])

        # decay on counters
        self.model[state, action] = [state_, reward]
        temp = self.counter.copy()

        for k,v in temp.items():
            self.counter.update({k:v*self.lamb_da})
            self.N_counter.update({k:self.N_counter[k]*self.lamb_da})

        # model planning
        N = self.dyna_N
        for i in range(N):
            rs, ra = random.choice([k for k,v in self.model.items()])
            rs_, rr = self.model[(rs, ra)]

            self.q_table[rs][ra] += self.alpha*(rr
               +self.gamma*max([(self.q_table[rs_][i]-self.N_counter[state_][i]/self.k) for i in range(4)])
               -self.q_table[rs][ra])

            # self.q_table[rs][ra] += self.alpha*(rr
            #    +self.gamma*max([(self.q_table[rs_][i]+1/self.N_counter[state_][i]) for i in range(4)])
            #    -self.q_table[rs][ra])

            # self.q_table[rs][ra] += self.alpha*(rr
            #                                        +self.gamma*max(self.q_table[rs_])
            #                                        -self.q_table[rs][ra])



        return self.q_table

    def get_path(self,s):
        if len(s)==4:
            s.append(False)
        state = tuple(s)

        return self.q_table[state].index(max(self.q_table[state]))

    ### END CODE HERE ###
