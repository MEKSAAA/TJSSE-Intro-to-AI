# DQN in LunarLander-v2

#### 作者：2253893 苗君文

该项目实现了一个深度Q网络（DQN），以解决OpenAI Gym中的LunarLander-v2环境。

## 项目结构

- `main.py`：训练DQN代理的主脚本。
- `dqn_agent.py`：定义了与环境交互并从中学习的Agent类。
- `model.py`：定义了神经网络结构。
- `requirements.txt`：列出了所需的Python包。

## 环境设置和安装

### 前提条件

确保你已经安装了`conda`。

### 步骤说明
1. **克隆仓库**


2. **创建Conda环境**

   ```bash
   conda create --name lunar python=3.8
   conda activate lunar
   
3. **安装所需包**
   
   ```bash
   pip install -r requirements.txt

## 运行代码

### 训练智能体

要训练DQN智能体，只需运行main.py脚本：

   ```bash
   python main.py
   ``` 

这将开始训练过程，并且每100个episode打印一次平均得分。脚本还会将具有最佳平均得分的模型保存到best_checkpoint.pth。

代码中默认使用的是DQN网络，用户可以将dqn_agent.py中的以下代码取消注释，即可用Double DQN网络训练模型。

```bash
# Double DQN: use the local model to select the best action
best_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
Q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions)
```
   
### 可视化训练进度

训练脚本会生成一个显示每个episode得分的图表，展示智能体的学习进度。训练结束后，图表将自动显示。

图表中展示score, average score, solved requirement线，方便查看训练结果。

### 渲染智能体

训练结束后，可以通过运行main.py中的渲染部分观看训练后的智能体在LunarLander环境中的表现。脚本会加载最佳模型并渲染几次episode。

## 示例输出

训练期间，终端将输出类似以下的信息：

   ```plaintext
Episode 100	Average Score: -146.12
Episode 200	Average Score: -122.28
Episode 300	Average Score: -46.20
Episode 400	Average Score: -49.59
Episode 500	Average Score: -40.01
Episode 600	Average Score: -22.04
Episode 700	Average Score: 25.93
Episode 800	Average Score: 145.30
Episode 808	Average Score: 151.50
Average Score exceeded 100! Enabling rendering...
Episode 900	Average Score: 172.63
Episode 1000	Average Score: 145.97
Episode 1100	Average Score: 177.34
Episode 1200	Average Score: 184.97
Episode 1300	Average Score: 179.79
Episode 1372	Average Score: 200.44
Environment solved in 1272 episodes!	Average Score: 200.44
   ```


生成的图表示例：

![LularLander Plot](.\image\plot_eg.png)


生成的渲染结果示例：

![LunarLander Render](.\image\lunarlander1.gif)