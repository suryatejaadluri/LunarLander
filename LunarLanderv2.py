import gym
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import random
from collections import deque
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Neural Network with two hidden layers
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

    def __init__(self, alpha = 0.0005, gamma = 0.99, epsilon = 1.0, n_eps = 2000, N = 70000, C = 1500, M = 32, seed = 6, save_flag = True):
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
        self.save_flag = save_flag

    def store_memory(self, state, action, reward, next_state, done):
        reward = np.array([reward], dtype=float)
        action = np.array([action], dtype=int)
        done = np.array([done], dtype=int)
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self, M):
        batch = np.array(random.sample(self.memory, k=M), dtype=object)
        batch = batch.T
        batch = batch.tolist()
        return (torch.tensor(batch[0]).to(device), torch.tensor(batch[1], dtype=torch.int64).to(device),
                torch.tensor(batch[2], dtype=torch.float).to(device), torch.tensor(batch[3]).to(device),
                torch.tensor(batch[4]).to(device))

    def solve(self):
        np.random.seed(self.seed)
        count = 0
        scores = []
        mean_scores = []
        recent_scores = deque(maxlen=100)
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

                #Select action E-Greedily and take a step and store in  memory
                if greed < self.epsilon:
                    action = np.random.randint(0, 4)
                else:
                    action = np.argmax(weights.detach().cpu().numpy())
                next_state, reward, done, data = self.env.step(action)
                score += reward
                self.store_memory(state, action, reward, next_state, done)

                #Populate buffer, if sufficiently populated sample from it
                if len(self.memory) < (7 * self.mini_batch_size):
                    state = deepcopy(next_state)
                    if done:
                        break
                    continue
                else:
                    transitions = self.sample_memory(self.mini_batch_size)

                #Optimize on the mini_batch using Adam
                states, actions, rewards, next_states, dones = transitions
                Q_t = self.Q_t(next_states).detach()
                Q_tmax = Q_t.max(1)[0].unsqueeze(1)
                y = rewards + (self.gamma * Q_tmax * (1-dones))
                Q = self.Q(states).gather(1, actions)
                loss = F.mse_loss(Q, y)
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
            recent_scores.append(score)
            scores.append(score)
            mean_scores.append(np.mean(recent_scores))
            #Decay epsilon at the end of the episode
            self.epsilon = max(0.1, 0.99 * self.epsilon)
            if np.mean(recent_scores) >= 225.0:
                if self.save_flag:
                  torch.save(self.Q.state_dict(), 'trainv1.pth')
                  break
        return scores,mean_scores

#alpha,gamma,epsilon,n_eps,N,C,M,seed
def graph1():
    agent = QLearningAgent(0.0005, 0.99, 1.0, 2000, 70000, 1500, 32, 6)
    scores,mean_scores = agent.solve()
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(scores)), mean_scores)
    plt.title('Fig 1. Performance of an DQN agent while training')
    plt.legend(['Scores', 'Mean Reward last 100 episodes:{:.2f}'.format(mean_scores[-1])], loc='lower right')
    plt.ylabel('Score')
    plt.xlabel('Episodes')
    plt.show()


def graph2():
    agent = QLearningAgent()
    agent.Q.load_state_dict(torch.load('trainv1.pth'))
    scores = []
    mean_scores = []
    #Setting different seed to test the agent
    agent.env.seed(420)

    for i in range(100):
        state = agent.env.reset()
        score = 0
        for j in range(1000):
            agent.Q.eval()
            with torch.no_grad():
                weights = agent.Q(torch.tensor(state).to(device))
            agent.Q.train()
            action = np.argmax(weights.detach().cpu().numpy())
            state, reward, done, _ = agent.env.step(action)
            agent.env.render()
            score += reward
            if done:
                break
        scores.append(score)
        mean_scores.append(np.mean(scores))

    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(scores)), mean_scores)
    plt.title('Fig 2. Performance of a trained agent for 100 episodes')
    plt.legend(['Scores', 'Mean Reward last 100 episodes:{:.2f}'.format(mean_scores[-1])], loc='lower right')
    plt.ylabel('Score')
    plt.xlabel('Episodes')
    plt.show()

def graph3():
    buffer_sizes = [2000,10000,70000,200000]
    scores_array = []
    mean_scores_array = []
    for i in range(len(buffer_sizes)):
        agent = QLearningAgent(0.0005, 0.99, 1.0, 1000, buffer_sizes[i], 1500, 32, 6, False)
        scores,mean_scores = agent.solve()
        scores_array.append(scores)
        mean_scores_array.append(mean_scores)
    plt.plot(np.arange(len(mean_scores_array[0])), mean_scores_array[0])
    plt.plot(np.arange(len(mean_scores_array[1])), mean_scores_array[1])
    plt.plot(np.arange(len(mean_scores_array[2])), mean_scores_array[2])
    plt.plot(np.arange(len(mean_scores_array[3])), mean_scores_array[3])
    plt.title('Fig 3. Training of the DQN agent for different replay buffersizes')
    plt.legend(['2K','10K','70K','200K'], loc='upper left')
    plt.ylabel('Mean Score for recent 100 episodes')
    plt.xlabel('No. of Episodes')
    plt.ylim(-260,260)
    plt.show()

def graph4():
    Learning_rates = [0.1,0.005,0.0005,0.00001]
    scores_array = []
    mean_scores_array = []
    for i in range(len(Learning_rates)):
        agent = QLearningAgent(Learning_rates[i], 0.99, 1.0, 1000,70000, 1500, 32, 6, False)
        scores,mean_scores = agent.solve()
        scores_array.append(scores)
        mean_scores_array.append(mean_scores)
    plt.plot(np.arange(len(mean_scores_array[0])), mean_scores_array[0])
    plt.plot(np.arange(len(mean_scores_array[1])), mean_scores_array[1])
    plt.plot(np.arange(len(mean_scores_array[2])), mean_scores_array[2])
    plt.plot(np.arange(len(mean_scores_array[3])), mean_scores_array[3])
    plt.title('Fig 4. Training of the DQN agent with different learning rates')
    plt.legend(['0.1','0.005','0.0005','0.00001'], loc='upper left')
    plt.ylabel('Mean Score for recent 100 episodes')
    plt.xlabel('No. of Episodes')
    plt.show()

def graph5():
    C_values= [50,1500,10000,15000]
    scores_array = []
    mean_scores_array = []
    for i in range(len(C_values)):
        agent = QLearningAgent(0.0005, 0.99, 1.0, 1000, 70000, C_values[i], 32, 6, False)
        scores,mean_scores = agent.solve()
        scores_array.append(scores)
        mean_scores_array.append(mean_scores)
    plt.plot(np.arange(len(mean_scores_array[0])), mean_scores_array[0])
    plt.plot(np.arange(len(mean_scores_array[1])), mean_scores_array[1])
    plt.plot(np.arange(len(mean_scores_array[2])), mean_scores_array[2])
    plt.plot(np.arange(len(mean_scores_array[3])), mean_scores_array[3])
    plt.title('Fig 5. Training of the DQN agent with different target update frequencies')
    plt.legend(['50','1500','10000','15000'], loc='upper left')
    plt.ylabel('Mean Score for recent 100 episodes')
    plt.xlabel('No. of Episodes')
    plt.show()

if __name__ == '__main__':
    globals()[sys.argv[1]]()

# graph2()