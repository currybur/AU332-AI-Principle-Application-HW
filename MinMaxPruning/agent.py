import random, re, datetime
from queue import PriorityQueue
import time

class Agent(object):
    def __init__(self, game):
        self.game = game

    def getAction(self, state):
        raise Exception("Not implemented yet")


class RandomAgent(Agent):
    def getAction(self, state):
        legal_actions = self.game.actions(state)
        self.action = random.choice(legal_actions)


class SimpleGreedyAgent(Agent):
    # a one-step-lookahead greedy agent that returns action with max vertical advance
    def getAction(self, state):
        legal_actions = self.game.actions(state)

        self.action = random.choice(legal_actions)
        player = self.game.player(state)
        if player == 1:
            max_vertical_advance_one_step = max([action[0][0] - action[1][0] for action in legal_actions])
            max_actions = [action for action in legal_actions if
                           action[0][0] - action[1][0] == max_vertical_advance_one_step]
        else:
            max_vertical_advance_one_step = max([action[1][0] - action[0][0] for action in legal_actions])
            max_actions = [action for action in legal_actions if
                           action[1][0] - action[0][0] == max_vertical_advance_one_step]
        self.action = random.choice(max_actions)


class MyTeam(Agent):


    def getAction(self, state):
        legal_actions = self.game.actions(state)
        self.action = random.choice(legal_actions)

        player = self.game.player(state)
        ### START CODE HERE ###

        depth = 1
        max_step = 30
        alpha = float('-inf')
        beta = float('inf')
        action_queue = PriorityQueue()  # search with bounded memory and preference
        update_queue = PriorityQueue()

        for action in legal_actions:
            action_queue.put((-(3-2*player)*(action[0][0] - action[1][0]),action))  # all actions sorted by vertical displacement

        start = time.time()
        while True:
            now = time.time()
            if now-start>=0.2:
                #print("depth",depth)
                break
            count = 0

            while (not action_queue.empty()) and max_step > count:
                action = action_queue.get()[1]
                count += 1

                if self.cal_value(self.game.succ(state, action), player) == 10086:  # evaluate after-action state
                    self.action = action
                    break
                #print(count)
                child_value = self.min_op(player, self.game.succ(state, action), depth, alpha, beta, max_step)

                update_queue.put((-child_value,action))
                if child_value > alpha:
                    alpha = child_value
                    self.action = action

            depth += 1

            #now = time.time()
            #if now-start>=1:
             #   break

            while not action_queue.empty():
                action_queue.get()

            while not update_queue.empty():
                action_queue.put(update_queue.get())

        #print(time.time()-start,depth)

    def cal_value(self, state, player):
        """
        evaluates the state, if win, 1000; if lose, -1000; else, a value(larger=better).
        :param state:
        :param player:
        :return:
        """

        board = state[1]

        player_pieces_position = board.getPlayerPiecePositions(player)
        enemy_pieces_position = board.getPlayerPiecePositions(3-player)

        player_vertical = 0
        for position in player_pieces_position:
            player_vertical += position[0]

        enemy_vertical = 0
        for position in enemy_pieces_position:
            enemy_vertical += position[0]

        player_horizontal = 0
        for position in player_pieces_position:
            player_horizontal += abs(abs(position[1] - min(position[0],20-position[0])/2)-1)

        enemy_horizontal = 0
        for position in enemy_pieces_position:
            enemy_horizontal += abs(abs(position[1] - min(position[0],20-position[0])/2)-1)

        if player == 1:
            if player_vertical == 30:   # the state is win ending
                return 10086
            if enemy_vertical == 170:   # lose ending
                return -10086
            else:
                return 400-(player_vertical + enemy_vertical)+(enemy_horizontal - player_horizontal)/2
            
        else:
            if player_vertical == 170:
                return 10086
            if enemy_vertical == 30:
                return -10086
            else:
                return (player_vertical + enemy_vertical)+(enemy_horizontal - player_horizontal)/2

    def maxi_op(self, player, state, depth, alpha, beta, max_step):
        
        if depth == 0:
            return self.cal_value(state, player)

        if self.cal_value(state, player) == -10086:
            return -10086
        
        depth -= 1
        node_value = float('-inf')
        action_queue = PriorityQueue()
        for action in self.game.actions(state):
            action_queue.put((-(3-2*player)*(action[0][0] - action[1][0]),action))

        count = 0
        while (not action_queue.empty()) and count < max_step:
            action = action_queue.get()[1]
            count += 1

            node_value = max(node_value, self.min_op(player, self.game.succ(state, action), depth, alpha, beta, max_step))
            if node_value >= beta:  # pruning
                return node_value
            alpha = max(alpha, node_value)
        return node_value

    def min_op(self, player, state, depth, alpha, beta, max_step):
        if depth == 0:
           return self.cal_value(state, player)
        if self.cal_value(state, player) == 10086:
           return 10086
        depth -= 1
        node_value = float('inf')
        action_queue = PriorityQueue()
        for action in self.game.actions(state):
            action_queue.put(((3-2*player)*(action[0][0] - action[1][0]),action))#search from the worst state
        count = 0
        while (not action_queue.empty()) and count < max_step:
            action = action_queue.get()[1]
            count += 1
            node_value = min(node_value, self.maxi_op(player, self.game.succ(state, action), depth, alpha, beta, max_step))
            if node_value <= alpha:
                return node_value
            beta = min(beta, node_value)
        return node_value

        ### END CODE HERE ###


