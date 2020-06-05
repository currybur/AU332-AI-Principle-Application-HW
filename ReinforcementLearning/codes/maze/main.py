from agent import Agent, DQN_Agent
import time
import json
import sys

file = ""

def store_table(table):
    '''
    Stores q_table, can derive the policy
    :param table:
    :return:
    '''
    temp = table.copy()
    for k,v in temp.items():
        # print(k)
        table[" ".join(str(i) for i in list(k))] = table.pop(k)
    # print(table)
    jsObj = json.dumps(table)
    fileObject = open(file, 'w')
    fileObject.write(jsObj)
    fileObject.close()

def read_table():
    '''
    load q_values as policy
    :return:
    '''
    fileObject = open(file, 'r')
    jsObj = json.loads(fileObject.read())
    temp = jsObj.copy()
    for k, v in temp.items():
        l = []
        for i in k.split():
            if i != "True" and i != "False":
                l.append(float(i))
            else:
                l.append({"True":True, "False":False}[i])
        jsObj[tuple(l)] = jsObj.pop(k)
    return jsObj

def display_table(table):
    c = 0
    for k,v in table.items():
        c += 1
        print(v,"\t",end="")
        if c == 6:
            c = 0
            print()
    print()

def check_converge_time():
    '''
    to show how many episodes needed to find the optimal path
    for the first time
    :return:
    '''
    env = Maze()
    cvg_time = 0
    for i in range(100):
        print(i)
        agent = DQN_Agent(actions=list(range(env.n_actions)))
        flag = 0
        for episode in range(300):
            if flag:
                break
            s = env.reset()
            episode_reward = 0
            while True:
                #env.render()                 # You can comment all render() to turn off the graphical interface in training process to accelerate your code.
                a = agent.choose_action(s)
                s_, r, done = env.step(a)
                q_table = agent.update_q(s, s_, a, r)
                episode_reward += r
                if episode_reward == 4:
                    cvg_time+=episode
                    flag = 1
                s = s_
                #print(s)
                if done:
                    #env.render()
                    #time.sleep(0.5)
                    break
            # print('episode:', episode, 'episode_reward:', episode_reward)
        if flag == 0:
            cvg_time += 300
    print(cvg_time/100)

def common_check(episodes = 400):
    '''
    an ordinary learning process, and store q_table
    :return:
    '''
    env = Maze()
    agent = DQN_Agent(actions=list(range(env.n_actions)))
    for episode in range(episodes):
        s = env.reset()
        episode_reward = 0
        while  True:
            #env.render()                 # You can comment all render() to turn off the graphical interface in training process to accelerate your code.
            a = agent.choose_action(s)
            s_, r, done = env.step(a)
            q_table = agent.update_q(s, s_, a, r)
            episode_reward += r
            s = s_
            if done:
                #env.render()
                break
        print('episode:', episode, 'episode_reward:', episode_reward)
    store_table(q_table)

def see_path():
    '''
    show the path from q_table
    :return:
    '''
    q_table = read_table()
    env = Maze()
    agent = DQN_Agent(actions=list(range(env.n_actions)))
    agent.load_q_table(q_table)
    s = env.reset()
    while  True:
        env.render()
        a = agent.get_path(s)
        # print(a)
        time.sleep(0.2)
        s_, r, done = env.step(a)
        s = s_
        if done:
            env.render()
            break

if __name__ == "__main__":
    ### START CODE HERE ###
    maze = '2'
    max_episode = 300
    if len(sys.argv)>1:
        if len(sys.argv)==2:
            maze = "{}".format(sys.argv[1])
        if len(sys.argv)==3:
            max_episode =sys.argv[2]

    
    filename = "table_maze_"

    
    if maze == '1':
        from maze_env1 import Maze
        file = filename+"1.json"
    elif maze == '2':
        file = filename+"2.json"
        from maze_env2 import Maze

    common_check(max_episode)
    # see_path()

    ### END CODE HERE ###

    print('\ntraining over\n')
