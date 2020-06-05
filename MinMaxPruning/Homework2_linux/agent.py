import random, re, datetime


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


class TeamNameMinimaxAgent(Agent):
    def getAction(self, state):
        legal_actions = self.game.actions(state)
        self.action = random.choice(legal_actions)

        player = self.game.player(state)
        ### START CODE HERE ###







        ### END CODE HERE ###


class myteam_new(Agent):


    def getAction(self, state):
        legal_actions = self.game.actions(state)
        self.action = random.choice(legal_actions)

        player = self.game.player(state)
        #implement minimax
        depth = 3
        memory = 20
        v = -float('inf')

        order = PriorityQueue()  # search with bounded memory and preference
        for action in legal_actions:
            order.put((-(3-2*player)*(action[0][0] - action[1][0]),action))  # all actions sorted by vertical displacement
        order_next = PriorityQueue()

        start = time.time()
        count = 0
        while (not order.empty()) and time.time()-start<0.99:
            action = order.get()[1]
            if self.eval(self.game.succ(state, action),player) == 10086:  # evaluate after-action state
                self.action = action
                break
            count += 1

            v_next = self.min_value(player,self.game,self.game.succ(state, action),depth,v,float('inf'),start)

            order_next.put((-v_next,action))
            if v_next > v:
                v = v_next
                self.action = action



    def eval(self,state,player):
        """
        evaluates the state, if win, 1000; if lose, -1000; else, a value(larger=better).
        :param state:
        :param player:
        :return:
        """

        board = state[1]
        #status = board.board_status #a dictionary that stores whole board infomation given a state
        player_status = board.getPlayerPiecePositions(player)  # pieces' positions of player
        opponent_status = board.getPlayerPiecePositions(3-player)  # pieces' positions of opponent

        player_vertical_count = -30
        for position in player_status:

            player_vertical_count += position[0]

        opponent_vertical_count = -30
        for position in opponent_status:
            opponent_vertical_count += position[0]

        player_horizontal_count = 0
        for position in player_status:
            player_horizontal_count += abs(abs(position[1] - (11-abs(position[0]-10))/2)-1) #1 -> 5.5 -> 1

        opponent_horizontal_count = 0
        for position in opponent_status:
            opponent_horizontal_count += abs(abs(position[1] - (11-abs(position[0]-10))/2)-1)

        if player == 1:
            if player_vertical_count == 0:  # player win!
                return 10086
            if opponent_vertical_count == 140:  # player lose!
                return -10086
            else:
                return 280-(player_vertical_count + opponent_vertical_count)+(opponent_horizontal_count - player_horizontal_count)/2

        else:
            if player_vertical_count == 140:
                return 10086
            if opponent_vertical_count == 0:
                return -10086
            else:
                return (player_vertical_count + opponent_vertical_count)+(opponent_horizontal_count - player_horizontal_count)/2
            #for player == 1, the player_vertical_count should be the less the better,
            # and opponent_vertical_count should be the less the better, too.
            #(Since that implies player is close to top but opponent is far from bottom.)
            #The maximum vertical_count == 20*gamesize - 30
            #The minimum vertical_count == 30

    def max_value(self,player,game,state,depth,a,b,start):

        if depth == 0:
            return self.eval(state,player)

        if self.eval(state,player) == -10086:
            return -10086

        depth -= 1
        v = -float('inf')
        order = PriorityQueue()
        for action in self.game.actions(state):
            order.put((-(3-2*player)*(action[0][0] - action[1][0]),action))

        count = 0
        while (not order.empty()) and time.time()-start <0.99:
            action = order.get()[1]
            count += 1
            #print("a",count)
            v = max(v, self.min_value(player,game,self.game.succ(state, action),depth,a,b,start))
            if v >= b:  # pruning
                return v
            a = max(a,v)
        return v

    def min_value(self,player,game,state,depth,a,b,start):

        if depth == 0:
           return self.eval(state,player)

        if self.eval(state,player) == 10086:
           return 10086

        depth -= 1
        v = float('inf')
        order = PriorityQueue()
        for action in self.game.actions(state):
            order.put(((3-2*player)*(action[0][0] - action[1][0]),action))#search from the worst state

        count = 0
        while (not order.empty()) and time.time()-start <0.99:
            action = order.get()[1]
            count += 1
            #print("i",count)
            v = min(v, self.max_value(player,game,self.game.succ(state, action),depth,a,b,start))
            if v <= a:
                return v
            b = min(b,v)
        return v


class MinimaxAgent(Agent):
    def getAction(self, state):
        legal_actions = self.game.actions(state)
        self.action = random.choice(legal_actions)

        player = self.game.player(state)
        ### START CODE HERE ###

        start_time = time.time()

        depth = 3
        alpha = float('-inf')
        beta = float('inf')

        action_queue = PriorityQueue()
        for action in legal_actions:
            action_queue.put((-(3-2*player)*(action[0][0] - action[1][0]), action))  # On the board, player 1 is at the bottom.  Since Priority Queue is ascending sorted and this is a maximum here, we add a minus sign.
        step=0
        while (not action_queue.empty()) and step<20:
            action = action_queue.get()[1]
            if self.cal_value(self.game.succ(state, action), player) == 10086:  # evaluate after-action state
                self.action = action
                break
            step+=1
            child_value = self.mini_operation(player, self.game.succ(state, action), depth, alpha, beta,start_time)

            if child_value > alpha:
                alpha = child_value
                self.action = action  # update action in time

    def cal_value1(self,state,player):

        board = state[1]

        player_pieces = board.getPlayerPiecePositions(player)  # pieces' positions of player
        enemy_pieces = board.getPlayerPiecePositions(3-player)  # pieces' positions of enemy

        player_vertical_heu = -30
        opponent_vertical_heu = -30

        for position in player_pieces:
            player_vertical_heu += position[0]

        for position in enemy_pieces:
            opponent_vertical_heu += position[0]

        player_horizontal_heu = 0
        for position in player_pieces:
            player_horizontal_heu += abs(abs(position[1] - (11-abs(position[0]-10))/2)-1) #1 -> 5.5 -> 1

        opponent_horizontal_heu = 0
        for position in enemy_pieces:
            opponent_horizontal_heu += abs(abs(position[1] - (11-abs(position[0]-10))/2)-1)

        if player == 1:
            if player_vertical_heu == 0:  # start from bottom, arrive at top
                return 10086
            if opponent_vertical_heu == 140:  # vise versa
                return -10086
            else:
                return 280-(player_vertical_heu + opponent_vertical_heu)+(opponent_horizontal_heu - player_horizontal_heu)/2

        else:
            if player_vertical_heu == 140:
                return 10086
            if opponent_vertical_heu == 0:
                return -10086
            else:
                return (player_vertical_heu + opponent_vertical_heu)+(opponent_horizontal_heu - player_horizontal_heu)/2

    def cal_value(self,state,player):
        """
        evaluates the state, if win, 1000; if lose, -1000; else, a value(larger=better).
        :param state:
        :param player:
        :return:
        """

        board = state[1]
        #status = board.board_status #a dictionary that stores whole board infomation given a state
        player_status = board.getPlayerPiecePositions(player)  # pieces' positions of player
        opponent_status = board.getPlayerPiecePositions(3-player)  # pieces' positions of opponent

        player_vertical_count = -30
        for position in player_status:

            player_vertical_count += position[0]

        opponent_vertical_count = -30
        for position in opponent_status:
            opponent_vertical_count += position[0]

        player_horizontal_count = 0
        for position in player_status:
            player_horizontal_count += abs(abs(position[1] - (11-abs(position[0]-10))/2)-1) #1 -> 5.5 -> 1

        opponent_horizontal_count = 0
        for position in opponent_status:
            opponent_horizontal_count += abs(abs(position[1] - (11-abs(position[0]-10))/2)-1)

        if player == 1:
            if player_vertical_count == 0:  # player win!
                return 10086
            if opponent_vertical_count == 140:  # player lose!
                return -10086
            else:
                return 280-(player_vertical_count + opponent_vertical_count)+(opponent_horizontal_count - player_horizontal_count)/2

        else:
            if player_vertical_count == 140:
                return 10086
            if opponent_vertical_count == 0:
                return -10086
            else:
                return (player_vertical_count + opponent_vertical_count)+(opponent_horizontal_count - player_horizontal_count)/2

    def mini_operation(self,player,state,depth,alpha,beta, start_time):
        if depth == 0:
            return self.cal_value(state,player)
        if self.cal_value(state, player)==10086:
            return 10086

        node_value=float('inf')  # since this is minimum node
        depth -= 1
        action_queue = PriorityQueue()
        for action in self.game.actions(state):
            action_queue.put(((3-2*player)*(action[0][0] - action[1][0]),action))  # since minimum node
        step = 0
        while (not action_queue.empty()) and step<20:
            step += 1
            print("i",step)
            action = action_queue.get()[1]
            node_value = min(node_value, self.max_operation(player, self.game.succ(state,action), depth, alpha, beta, start_time))
            if node_value <= alpha:  # prune
                return node_value
            beta = min(beta, node_value)
        return node_value

    def max_operation(self,player,state,depth,alpha,beta, start_time):
        if depth == 0:
            return self.cal_value(state,player)
        if self.cal_value(state,player)==-10086:
            return -10086

        node_value=float('-inf')  # since this is minimum node
        depth -= 1
        action_queue = PriorityQueue()
        for action in self.game.actions(state):
            action_queue.put( (-(3-2*player)*(action[0][0] - action[1][0]),action) )  # since minimum node
        step=0
        while (not action_queue.empty()) and step<20:
            step+=1
            print("a",step)
            action = action_queue.get()[1]
            node_value = max(node_value, self.mini_operation(player, self.game.succ(state,action), depth, alpha, beta, start_time))
            if node_value >= beta:
                return node_value
            alpha = max(alpha, node_value)
        return node_value


        ### END CODE HERE ###


class another(Agent):
    def getLength(self, action, player):
        # return the vertical length change after the action for the player
        if (player == 1):
            return action[0][0] - action[1][0]
        else:
            return action[1][0] - action[0][0]
    def evaluate(self, state):
        # evalutaion of the whole board status
        score = 0
        board = state[1]
        l1, l2 = 0, 100
        for (i, j) in self.allPos:
            if board.board_status[(i, j)] == 1:
                score += self.w1[19 - i]
                l1 = max(l1, i)
            if board.board_status[(i, j)] == 2:
                score -= self.w1[i - 1]
                l2 = min(l2, i)
        board = state[1]
        poss = board.getPlayerPiecePositions(1)
        for pos in poss:
            acts = board.getAllHopPositions(pos)
            l = 0
            for act in acts:
                l = max(l, self.getLength([pos, act], 1))
            score += self.w2[l]

        poss = board.getPlayerPiecePositions(2)
        for pos in poss:
            acts = board.getAllHopPositions(pos)
            l = 0
            for act in acts:
                l = max(l, self.getLength([pos, act], 1))
            score -= self.w2[l]

        cnt1 = 0
        for i in range(1, 5):
            for j in range(1, i + 1):
                if (board.board_status[(i, j)] == 1): cnt1 += 1

        cnt2 = 0
        for i in range(16, 20):
            for j in range(1, 20 - i + 1):
                if (board.board_status[(i, j)] == 2): cnt2 += 1

        score += (cnt1 - cnt2) * 200
        score += ((20+1*cnt1*cnt1) * (-l1) + (20+1*cnt2*cnt2) * (-l2)) * 2
        return score

    def MinMaxSearch(self, state, alpha, beta, depth):
        if (depth >= 2):
            return self.evaluate(state)

        player = self.game.player(state)
        legal_actions = self.game.actions(state)
        legal_actions = [action for action in legal_actions if self.getLength(action, player) >= 0]
        legal_actions = [[self.getLength(action,player), action] for action in legal_actions]
        legal_actions.sort(reverse=True)
        if (player == 1):
            v = -1e10
            for action2 in legal_actions:
                action = action2[1]
                v = max(v, self.MinMaxSearch(self.game.succ(state, action), alpha, beta, depth + 1))
                if v > alpha:
                    alpha = v
                    if depth == 0: self.action = action
                if alpha >= beta: return v
            return v
        else:
            v = 1e10
            for action2 in legal_actions:
                action = action2[1]
                v = min(v, self.MinMaxSearch(self.game.succ(state, action), alpha, beta, depth + 1))
                if v < beta:
                    beta = v
                    if depth == 0: self.action = action
                if alpha >= beta: return v
            return v

    def check(self, state, player):
        board = state[1]
        max1 = 0
        min2 = 1e9
        for (i, j) in self.allPos:
            if board.board_status[(i, j)] == 1 and max1 < i:
                max1 = i
            if board.board_status[(i, j)] == 2 and min2 > i:
                min2 = i
        if (max1>=min2): return 2
        if (player==1): return max1>5
        if (player==2): return min2<15

    def miss(self, state, player):
        board = state[1]
        cnt1, s11, s12, s21, s22 = 0, 0, 0, 0, 0
        for i in range(1, 5):
            for j in range(1, i + 1):
                if (board.board_status[(i, j)] == 1): cnt1 += 1
                s11 += i
                s12 += j

        cnt2 = 0
        for i in range(16, 20):
            for j in range(1, 20 - i + 1):
                if (board.board_status[(i, j)] == 2): cnt2 += 1
                s21 += i
                s22 += j

        if player == 1:
            if cnt2 == 10: return [-1, s11, s12]
            return [10 - cnt1, s11, s12]

        if player == 2:
            if cnt1 == 10: return [-1, s21, s22]
            return [10 - cnt2, s21, s22]

    def sum(self, state, player):
        board = state[1]
        s1, s2 = 0, 0
        for i in range(1, 11):
            for j in range(1, i + 1):
                if board.board_status[(i, j)] == 1 and player == 1:
                    s1 += i
                    s2 += j
                if board.board_status[(i, j)] == 2 and player == 2:
                    s1 += i
                    s2 += j
        for i in range(11, 20):
            for j in range(1, 21 - i):
                if board.board_status[(i, j)] == 1 and player == 1:
                    s1 += i
                    s2 += j
                if board.board_status[(i, j)] == 2 and player == 2:
                    s1 += i
                    s2 += j
        return [s1, s2]

    def dfs(self, state, player, depth, act):
        p = self.game.player(state)
        legal_actions = self.game.actions(state)
        legal_actions = [action for action in legal_actions if self.getLength(action, p) >= 0]
        [cnt, s1, s2] = self.miss(state, player)
        s11, s22 = self.sum(state, player)
        if cnt < 0: return
        if p != player:
            self.dfs(self.game.succ(state, legal_actions[0]), player, depth, act)
            return
        if (depth + cnt >= self.best):
            return

        if (cnt == 0):
            self.action = act
            self.best = depth
            print ('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
            print (self.best)
            return

        acts = [[abs(s11 + action[1][0]  - action[0][0] - s1) + abs(s22 + action[1][1]  - action[0][1] - s2), action] for action in legal_actions]
        acts.sort()
        for action2 in acts:
            action = action2[1]
            if depth == 0:
                act = action
            self.dfs(self.game.succ(state, action), player, depth + 1, act)

    def GreedySearch(self, state, player, depth, act, summ, lastact):
        p = self.game.player(state)
        legal_actions = self.game.actions(state)
        legal_actions = [action for action in legal_actions if self.getLength(action, p) >= 0]
        [cnt, s1, s2] = self.miss(state, player)
        if cnt < 0: return
        if p != player:
            self.GreedySearch(self.game.succ(state, legal_actions[0]), player, depth, act,summ,act)
            return
        acts = [[self.getLength(action,player), action] for action in legal_actions]
        acts.sort(reverse=True)
        if (len(acts)==0):return
        t=max(summ/depth,(summ+acts[0][0])/(depth+1))
        if t>self.best:
            self.best=t
            self.action=act
            print ('best is',self.best, depth+1)

        if (depth >= 3):
            return

        for action2 in acts:
            action = action2[1]
            if (action[0]!=lastact[1]):continue
            if depth == 0:
                act = action
            self.GreedySearch(self.game.succ(state, action), player, depth + 1, act,summ+action2[0],action)

    def dist(self, action, player):
        if player==1: return action[0][0]-1
        else: return 19-action[0][0]

    def getAction(self, state):
        self.allPos = []
        for i in range(1, 11):
            for j in range(1, i + 1):
                self.allPos.append((i, j))
        for i in range(11, 20):
            for j in range(1, 21 - i):
                self.allPos.append((i, j))
        self.w1 = [10*i for i in range(25)]
        self.w2 = [5*i for i in range(25)]

        legal_actions = self.game.actions(state)
        player = self.game.player(state)
        legal_actions = [action for action in legal_actions if self.getLength(action, player) >= 0]

        if player == 1:
            max_vertical_advance_one_step = max([action[0][0] - action[1][0] for action in legal_actions])
            max_actions = [action for action in legal_actions if
                           action[0][0] - action[1][0] == max_vertical_advance_one_step]
        else:
            max_vertical_advance_one_step = max([action[1][0] - action[0][0] for action in legal_actions])
            max_actions = [action for action in legal_actions if
                           action[1][0] - action[0][0] == max_vertical_advance_one_step]
        self.action = random.choice(max_actions)
        print (self.action)

        stage = self.check(state, player)
        print ('stage is ',stage)
        if stage==2:
            self.MinMaxSearch(state, -1e10, 1e10, 0)
        elif stage==1:
            self.best=0
            legal_actions = [action for action in legal_actions if self.dist(action,player)>= 4]
            for act in legal_actions:
                self.GreedySearch(self.game.succ(state, act), player, 1, act,self.getLength(act,player), act)
        else:
            self.best = 4
            self.dfs(state, player, 0, 1)
