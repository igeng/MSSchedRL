import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class MultiResourceAttention(nn.Module):

    def __init__(self, d_model):
        super(MultiResourceAttention, self).__init__()
        self.d_model = d_model
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = np.sqrt(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        # 可选：添加 dropout 提升泛化能力
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (B, N, R, d_model)  其中 R 表示资源维度数量（例如2）
        Q = self.q_linear(x)  # (B, N, R, d_model)
        K = self.k_linear(x)  # (B, N, R, d_model)
        V = self.v_linear(x)  # (B, N, R, d_model)
        # 计算注意力分数：对每个节点计算各资源维度间的关系
        # 形状：(B, N, R, R)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = self.softmax(attn_scores)
        attn_weights = self.dropout(attn_weights)
        # 加权求和得到新表示：(B, N, R, d_model)
        attn_output = torch.matmul(attn_weights, V)
        # 残差连接与 LayerNorm：对每个资源维度分别归一化
        output = self.layer_norm(x + attn_output)
        return output

class StateEncoder(nn.Module):

    def __init__(self, d_model, hidden_size, resource_len=30, instance_len=4):
        super(StateEncoder, self).__init__()
        self.resource_len = resource_len  # 固定30
        self.instance_len = instance_len  # 固定4
        self.num_nodes = resource_len // 2  # 每个节点2个维度，共15个节点
        self.d_model = d_model
        # 对每个资源（CPU、内存）进行单维度嵌入
        self.cpu_embed = nn.Linear(1, d_model)
        self.mem_embed = nn.Linear(1, d_model)
        # 使用 MultiResourceAttention 替代原有的单一注意力机制
        self.multi_resource_attn = MultiResourceAttention(d_model)
        # 融合各节点表示后，与微服务实例信息拼接
        # 输出维度： num_nodes * d_model + instance_len
        self.fc = nn.Linear(self.num_nodes * d_model + instance_len, hidden_size)

    def forward(self, state):
        # 如果输入状态为一维，则加上 batch 维度
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, 34)
        # state: (B, 34)
        resource = state[:, :self.resource_len]  # (B, 30)
        instance = state[:, self.resource_len:]    # (B, 4)
        batch_size = state.size(0)
        # 重塑为 (B, 15, 2) ：15 个节点，每个节点两个资源特征
        resource = resource.view(batch_size, self.num_nodes, 2)
        # 分离 CPU 与内存信息，并扩展维度至 (B, 15, 1)
        cpu = resource[:, :, 0].unsqueeze(-1)  # (B, 15, 1)
        mem = resource[:, :, 1].unsqueeze(-1)  # (B, 15, 1)
        # 分别嵌入到 d_model 空间，结果 (B, 15, d_model)
        cpu_emb = self.cpu_embed(cpu)
        mem_emb = self.mem_embed(mem)
        # 将两种资源嵌入堆叠： (B, 15, 2, d_model)
        resources = torch.stack([cpu_emb, mem_emb], dim=2)
        # 多资源注意力融合，输出形状依然 (B, 15, 2, d_model)
        fused_resources = self.multi_resource_attn(resources)
        # 对资源维度进行平均融合： (B, 15, d_model)
        fused = fused_resources.mean(dim=2)
        # 展平节点表示：(B, 15*d_model)
        fused_flat = fused.view(batch_size, -1)
        # 拼接微服务实例信息，得到 (B, 15*d_model + 4)
        concat = torch.cat([fused_flat, instance], dim=1)
        out = F.relu(self.fc(concat))
        return out

class Actor(nn.Module):
    """Actor (Policy) Model with MultiResourceAttention for state encoding."""
    def __init__(self, state_size, action_size, hidden_size=256, d_model=32):

        super(Actor, self).__init__()
        self.state_encoder = StateEncoder(d_model=d_model, hidden_size=hidden_size, resource_len=30, instance_len=4)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.state_encoder(state)  # 得到隐藏层表征
        x = F.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs

    def evaluate(self, state, epsilon=1e-6):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # 处理概率为0的情况，避免 log(0)
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities

    def get_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu()

    def get_det_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        return action.detach().cpu()

class Critic(nn.Module):
    """Critic (Value) Model with MultiResourceAttention for state encoding."""
    def __init__(self, state_size, action_size, hidden_size=256, seed=1, d_model=32):

        super(Critic, self).__init__()
        self.state_encoder = StateEncoder(d_model=d_model, hidden_size=hidden_size, resource_len=30, instance_len=4)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.seed = torch.manual_seed(seed)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        # state: (B, 34) 或 (34,)
        x = self.state_encoder(state)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
