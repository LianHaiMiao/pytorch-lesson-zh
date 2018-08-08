import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyGradient:
    # 初始化时, 我们需要给出这些参数, 并创建一个神经网络.
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.95):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # reward 逐步递减
        self.ep_obs, self.ep_acs, self.ep_res = [], [], [] # 分别存储每一步的observation, action 和 reward
        self.pnet = PGNet(n_features, n_actions) # 建立 policy network
        self.optimizer = torch.optim.Adam(self.pnet.parameters(), lr=learning_rate) # 建立优化器
        self.DEFAULT_TYPE = torch.float

    # 根据环境选择行为; 因为是 Policy Gradient 所以是按照概率从大到小去选择。
    def choose_action(self, observation):
        observation = self.to_tensor(observation, type=torch.float32)
        act_prob = self.pnet.get_act_prob(observation)
        act_prob = self.tesnsor2np(act_prob)
        action = np.random.choice(range(self.n_actions), p=act_prob)
        return action

    # 存储每一个回合的observation, action 和 reward, 当每次模拟结束之后,清空列表
    def store_transition(self, ob, a, r):
        self.ep_obs.append(ob)
        self.ep_acs.append(a)
        self.ep_res.append(r)
    # 学习参数更新
    def learn(self):
        # 衰减, 并标准化这回合的 reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        # 变成tensor
        observations = self.to_tensor(self.ep_obs, type=torch.float)
        actions = self.to_tensor(self.ep_acs, type=torch.long)
        # 开始训练,进行梯度下降
        loss = self.pnet(actions, observations, discounted_ep_rs_norm)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 训练完毕,清空这一回合的data
        self.ep_obs, self.ep_acs, self.ep_res = [], [], []
        return discounted_ep_rs_norm  # 返回这一回合的 state-action value

    # reward 的逐步衰减
    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_res = np.zeros_like(self.ep_res)
        running_add = 0
        for t in reversed(range(0, len(self.ep_res))):
            running_add = running_add * self.gamma + self.ep_res[t]
            discounted_ep_res[t] = running_add

        # normalize episode rewards
        discounted_ep_res -= np.mean(discounted_ep_res)
        discounted_ep_res /= np.std(discounted_ep_res)
        dis_ep_res = torch.from_numpy(discounted_ep_res).to(dtype=self.DEFAULT_TYPE)
        return dis_ep_res
    # 列表或者单个数值变成tensor
    def to_tensor(self, l, type):
        ten = torch.FloatTensor(l)
        return ten.to(dtype=type)

    def tesnsor2np(self, tens):
        return tens.detach().numpy()

# 用于决策的网络
class PGNet(nn.Module):
    def __init__(self, num_fea, num_act):
        super(PGNet, self).__init__()
        # 第一层
        self.fc1 = nn.Linear(num_fea, 10)
        # 第二层
        self.out = nn.Linear(10, num_act)
        # initial
        self.fc1.weight.data.normal_(0, 0.3)  # initial fc1 weight
        self.out.weight.data.normal_(0, 0.3)  # initial out weight
        self.neg_log = nn.CrossEntropyLoss(reduce=False)
    def forward(self, acs, obs, res):
        x = self.fc1(obs)
        x = F.tanh(x)
        x = self.out(x)
        neg_log_prob = self.neg_log(x, acs)
        # 最大化 总体 reward (log_p * R) 就是在最小化 -(log_p * R)
        # neg_log_prob = F.cross_entropy(acs, res, reduce=False)
        # (res = 原本的rewards + 衰减的未来reward) 引导参数的梯度下降
        # loss = torch.mean(neg_log_prob * res)
        loss = torch.mean(neg_log_prob * res)
        return loss

    def get_act_prob(self, obs):
        x = self.fc1(obs)
        x = F.tanh(x)
        act_prob = F.softmax(self.out(x), dim=0)
        return act_prob



if __name__ == '__main__':
    # 对于 Policy Gradient 来说，我们是在它们每次模拟结束之后，对 policy network 进行更新
    # 结束一次模拟，就更新一次。并不会在模拟的过程中进行更新。

    RENDER = False  # 在屏幕上显示模拟窗口会拖慢运行速度, 我们等计算机学得差不多了再显示模拟
    DISPLAY_REWARD_THRESHOLD = 400  # 当回合总 reward 大于 400 时显示模拟窗口

    env = gym.make('CartPole-v0')   #  构建 CartPole 环境
    env = env.unwrapped  # 取消限制
    env.seed(1)     # 普通的 Policy gradient 方法, 使得回合的 variance 比较大, 所以我们选了一个好点的随机种子

    print(env.action_space.n)     # 显示可用 action
    print(env.observation_space)    # 显示可用 state 的 observation
    print(env.observation_space.high)   # 显示 observation 最高值
    print(env.observation_space.low)    # 显示 observation 最低值

    # 定义一个Policy Graident
    RL = PolicyGradient(
        n_actions =  env.action_space.n,  # 定义在一个 Policy 中，我们可以采取的策略有哪些
        n_features = env.observation_space.shape[0],  # 定义我们在实验过程中可以观测到的值的形式
        learning_rate = 0.02, # 定义学习速率
        reward_decay = 0.99  # 奖励逐步减少
    )

    # 定义主函数，开始进行模拟

    for i_episode in range(1000):
        observation = env.reset()  # 把环境重置到初始状态
        # 不死亡，不重新开始
        while True:
            if RENDER:
                env.render()  # 函数用于渲染出当前的智能体以及环境的状态
            # 根据当前的观察，选择一个合适的action
            action = RL.choose_action(observation)
            # 采取完action之后返回四个值
            # observation_(action之后环境的状态)、reward(当前Action即时奖励)
            # done(任务是否结束标记，True，reset任务)、info(额外诊断信息)
            observation_, reward, done, info = env.step(action)
            # 存储这一回合的 transition
            RL.store_transition(observation, action, reward)

            # 如果当前模拟结束
            if done:
                # 计算当前模拟整个过程的reward
                ep_rs_sum = sum(RL.ep_res)
                # 如果当前变量中没有running_reward
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                # 当reward足够的时候，也就是action被训练的差不多的时候，我们渲染每次的结果
                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True

                print("episode:", i_episode, "  reward:", int(running_reward))
                # 学习, 输出 vt
                vt = RL.learn()

                if i_episode % 100 == 0:
                    plt.plot(vt)    # plot 这个回合的 vt
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()
                break

            # 如果游戏没有结束，就继续
            observation = observation_

