import gym
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import random
from statistics import mean
from collections import deque
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Network(nn.Module):
    def __init__(self, seed):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 4)

    def forward(self, state):
        h = F.relu(self.fc1(state))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        y = self.fc4(h)
        return y

class QLearningAgent(object):
    def __init__(self, alpha, gamma, epsilon, n_eps, N, C, M, seed):
        self.memory = deque(maxlen=N)
        self.memory_max = N
        self.target_update = C
        self.Q_t = Network(seed).to(device)
        self.Q = Network(seed).to(device)
        self.alpha = alpha
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.alpha)
        self.gamma = gamma
        self.epsilon = epsilon
        self.seed = seed
        self.n_eps = n_eps
        self.mini_batch_size = M
        self.env = gym.make('LunarLander-v2')
        self.env.seed(seed)

    def store_memory(self, state, action, reward, next_state, done):
        reward = np.array([reward], dtype=float)
        action = np.array([action], dtype=int)
        done = np.array([done], dtype=int)
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self, M):
        batch = np.array(random.sample(self.memory, k=M), dtype=object)
        batch = batch.T
        batch = batch.tolist()
        #         print(batch[1])
        return (torch.tensor(batch[0]).to(device), torch.tensor(batch[1], dtype=torch.int64).to(device),
                torch.tensor(batch[2], dtype=torch.float).to(device), torch.tensor(batch[3]).to(device),
                torch.tensor(batch[4]).to(device))

    def solve(self):
        states = self.env.observation_space
        actions = self.env.action_space
        np.random.seed(self.seed)
        count = 0
        scores = []
        scores_window = deque(maxlen=100)
        for eps in range(self.n_eps):
            state = self.env.reset()
            score = 0
            for i in range(1000):
                greed = np.random.random()
                # Feed Forward once to predict the best action for current state
                self.Q.eval()
                with torch.no_grad():
                    weights = self.Q(torch.tensor(state).to(device))
                self.Q.train()
                if greed < self.epsilon:
                    action = np.random.randint(0, 4)
                else:
                    action = np.argmax(weights.detach().cpu().numpy())
                next_state, reward, done, data = self.env.step(action)
                # self.env.render()
                score += reward
                self.store_memory(state, action, reward, next_state, done)

                if len(self.memory) < (7 * self.mini_batch_size):
                    state = deepcopy(next_state)
                    if done:
                        break
                    continue
                else:
                    transitions = self.sample_memory(self.mini_batch_size)

                states, actions, rewards, next_states, dones = transitions
                Q_t = self.Q_t(next_states).detach()
                Q_tmax = Q_t.max(1)[0].unsqueeze(1)
                # case2 = rewards + self.gamma * Q_tmax
                # case1 = rewards
                y = rewards + (self.gamma * Q_tmax * (1-dones))
                # y = torch.where(dones < 1, case2, case1)
                Q = self.Q(states).gather(1, actions)
                # print(y)
                # print(Q)
                loss = F.mse_loss(Q, y)
                # print(loss)
                #print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                count += 1
                if count == self.target_update:
                    count = 0
                    #self.Q_t = deepcopy(self.Q)
                    self.Q_t.load_state_dict(self.Q.state_dict())
                state = deepcopy(next_state)
                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            self.epsilon = max(0.1, 0.995 * self.epsilon)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(eps, np.mean(scores_window)), end="")
            if eps % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(eps, np.mean(scores_window)))
            if np.mean(scores_window) >= 210.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(eps,
                                                                                          np.mean(scores_window)))
                torch.save(self.Q.state_dict(), 'train2.pth')
                break
        return scores
#alpha,gamma,epsilon,n_eps,N,C,M,seed
def graph1():
    agent = QLearningAgent(0.0005, 0.99, 1.0, 2000, 50000, 1000, 32, 6)
    scores = agent.solve()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episodes')
    plt.show()
# agent = QLearningAgent(0.001,0.99,1.0,2000,300000,5,64,5)
# agent.solve()
# agent.solve()

# agent.Q.load_state_dict(torch.load('checkpoint.pth'))

def graph2():
    agent = QLearningAgent(0.0005, 0.99, 1.0, 2000, 30000, 50, 128, 5)
    agent.Q.load_state_dict(torch.load('train2.pth'))
    scores = []
    for i in range(100):
        state = agent.env.reset()
        score = 0
        for j in range(1000):
            agent.Q.eval()
            with torch.no_grad():
                weights = agent.Q(torch.tensor(state).to(device))
            agent.Q.train()
            action = np.argmax(weights.detach().cpu().numpy())
            # agent.env.render()
            state, reward, done, _ = agent.env.step(action)
            # agent.env.render()
            score += reward
            if done:
                break
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores)), end="")

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episodes')
    plt.show()
graph1()
graph2()
