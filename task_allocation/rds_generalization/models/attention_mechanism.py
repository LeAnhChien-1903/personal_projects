import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class AttentionNetwork(nn.Module):
    def __init__(self, selected_size: int = 3, robot_size: int = 3, task_size: int = 6):
        super(AttentionNetwork, self).__init__()
        self.selected1 = nn.Linear(in_features= selected_size, out_features= 16)
        torch.nn.init.xavier_uniform_(self.selected1.weight) 
        self.selected2 = nn.Linear(in_features= 16, out_features= 16)
        torch.nn.init.xavier_uniform_(self.selected2.weight) 
        self.robot1 = nn.Linear(in_features= robot_size, out_features= 16)
        torch.nn.init.xavier_uniform_(self.robot1.weight) 
        self.robot2 = nn.Linear(in_features= 16, out_features= 16)
        torch.nn.init.xavier_uniform_(self.robot2.weight) 
        self.robot3 = nn.Linear(in_features= 16, out_features= 16)
        torch.nn.init.xavier_uniform_(self.robot3.weight)
        self.robot4 = nn.Linear(in_features= 16, out_features= 1)
        torch.nn.init.xavier_uniform_(self.robot4.weight)
        self.task1 = nn.Linear(in_features= task_size, out_features= 16)
        torch.nn.init.xavier_uniform_(self.task1.weight)
        self.task2 = nn.Linear(in_features= 16, out_features= 16)
        torch.nn.init.xavier_uniform_(self.task2.weight) 
        self.task3 = nn.Linear(in_features= 16, out_features= 16)
        torch.nn.init.xavier_uniform_(self.task3.weight)
        self.task4 = nn.Linear(in_features= 16, out_features= 1)
        torch.nn.init.xavier_uniform_(self.task4.weight)
        self.actor_out1 = nn.Linear(in_features= 64, out_features= 8)
        torch.nn.init.xavier_uniform_(self.actor_out1.weight)
        self.actor_out2 = nn.Linear(in_features= 8, out_features= 1)
        torch.nn.init.xavier_uniform_(self.actor_out2.weight)
        self.critic_out1 = nn.Linear(in_features= 64, out_features= 8)
        torch.nn.init.xavier_uniform_(self.critic_out1.weight)
        self.critic_out2 = nn.Linear(in_features= 8, out_features= 1)
        torch.nn.init.xavier_uniform_(self.critic_out2.weight)
        
    def forward(self, selected: torch.Tensor, robot: torch.Tensor, task: torch.Tensor):
        selected_feat = self.selected2(F.relu(self.selected1(selected)))
        
        robot_feat = self.robot2(F.relu(self.robot1(robot)))
        robot_weight = F.sigmoid(self.robot4(F.tanh(self.robot3(robot_feat))))
        
        task_feat = self.task2(F.relu(self.task1(task)))
        task_weight = F.sigmoid(self.task4(F.tanh(self.task3(task_feat))))
        
        robot_vec = torch.sum(robot_weight * robot_feat, dim= 1)
        task_vec = torch.sum(task_weight * task_feat, dim= 1)
        
        global_feat = torch.cat((selected_feat, robot_vec, task_vec), dim= 1)
        global_feat = global_feat.reshape(global_feat.shape[0], 1, global_feat.shape[1])
        
        global_feats = global_feat.expand(global_feat.shape[0], task_feat.shape[1], global_feat.shape[-1])
        global_local_feats = torch.cat((task_feat, global_feats), dim= -1)
        
        return global_local_feats
    
    def getProbability(self, global_local_feats: torch.Tensor):
        return F.softmax(self.actor_out2(F.relu(self.actor_out1(global_local_feats))), dim=1).reshape(global_local_feats.shape[0], global_local_feats.shape[1])
    
    def getValue(self, global_local_feats: torch.Tensor):
        return torch.mean(self.critic_out2(F.relu(self.critic_out1(global_local_feats))), dim=1)
    
    def getLastValue(self, selected: torch.Tensor, robot:torch.Tensor, task: torch.Tensor):
        global_local_feats = self.forward(selected, robot, task)
        value = self.getValue(global_local_feats)
        
        return value
    
    def getActionForTest(self, selected: torch.Tensor, robot:torch.Tensor, task: torch.Tensor):
        global_local_feats = self.forward(selected, robot, task)
        
        probs = self.getProbability(global_local_feats)
        
        return probs.argmax(dim = 1)
    
    def getAction(self, selected: torch.Tensor, robot:torch.Tensor, task: torch.Tensor):
        global_local_feats = self.forward(selected, robot, task)
        
        probs = self.getProbability(global_local_feats)
        value = self.getValue(global_local_feats)
        
        dist = Categorical(probs)
        
        action = dist.sample()
        
        return action, value, dist.log_prob(action), dist.entropy()
    
    def evaluateAction(self, selected: torch.Tensor, robot: torch.Tensor, task: torch.Tensor, action: torch.Tensor):
        global_local_feats = self.forward(selected, robot, task)
        probs = self.getProbability(global_local_feats)
        value = self.getValue(global_local_feats)
        
        dist = Categorical(probs)
        
        return value, dist.log_prob(action), dist.entropy()

if __name__ == '__main__':
    actor = AttentionNetwork()
    selected = torch.randn(5, 3)
    robot = torch.randn(5, 10, 3)
    task = torch.randn(5, 10, 9)
    action, value, log_prob, entropy = actor.getAction(selected, robot, task)
    print(action, value, log_prob, entropy)
    print(actor.evaluateAction(selected, robot, task, action))