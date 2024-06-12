from math import e
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class PlanningNetwork(nn.Module):
    def __init__(self, robot_vec_size: int = 5, next_point_vec_size: int = 4, graph_vec_size: int = 8):
        super(PlanningNetwork, self).__init__()
        self.graph1 = nn.Linear(in_features= graph_vec_size, out_features= 16)
        self.graph2 = nn.Linear(in_features= 16, out_features= 16)
        self.graph3 = nn.Linear(in_features= 16, out_features= 16)
        self.graph4 = nn.Linear(in_features= 16, out_features= 1)
        
        self.robot1 = nn.Linear(in_features= robot_vec_size, out_features= 16)
        self.robot2 = nn.Linear(in_features= 16, out_features= 16)
        
        self.next1 = nn.Linear(in_features= next_point_vec_size, out_features= 16)
        self.next2 = nn.Linear(in_features= 16, out_features= 16)
        self.next3 = nn.Linear(in_features= 16, out_features= 16)
        self.next4 = nn.Linear(in_features= 16, out_features= 1)
        
        self.graph5 = nn.Linear(in_features= 16, out_features= 32)
        self.graph6 = nn.Linear(in_features= 32, out_features= 32)
        self.graph7 = nn.Linear(in_features= 32, out_features= 32)
        self.graph8 = nn.Linear(in_features= 32, out_features= 1)
        
        self.actor_out1 = nn.Linear(in_features= 80, out_features= 16)
        self.actor_out2 = nn.Linear(in_features= 16, out_features= 1)
        
        self.critic_out1 = nn.Linear(in_features= 80, out_features= 16)
        self.critic_out2 = nn.Linear(in_features= 16, out_features= 1)
        
    def graphForward(self, graph_data: torch.Tensor):
        graph_out1 = F.relu(self.graph1(graph_data))  
        graph_out2 = self.graph2(graph_out1)  
        graph_out3 = F.tanh(self.graph3(graph_out2))
        graph_out4 = F.sigmoid(self.graph4(graph_out3))  
        
        return torch.sum(graph_out4 * graph_out2, dim= 1)
    def forward(self, robot_data: torch.Tensor, next_point_data: torch.Tensor, graph_data: torch.Tensor):
        graph_out = self.graphForward(graph_data)
        robot_feat = self.robot2(F.relu(self.robot1(robot_data)))
        
        next_feat = self.next2(F.relu(self.next1(next_point_data)))
        next_weight = F.sigmoid(self.next4(F.tanh(self.next3(next_feat))))
        
        graph_feat = self.graph6(F.relu(self.graph5(graph_out)))
        graph_weight = F.sigmoid(self.graph8(F.tanh(self.graph7(graph_feat))))
        
        next_vec = torch.sum(next_weight * next_feat, dim= 1)
        graph_vec = torch.sum(graph_weight * graph_feat, dim= 1)
        
        global_feat = torch.cat((robot_feat, graph_vec, next_vec), dim= 1)
        global_feat = global_feat.reshape(global_feat.shape[0], 1, global_feat.shape[1])
        
        global_feats = global_feat.expand(global_feat.shape[0], next_feat.shape[1], global_feat.shape[-1])
        global_local_feats = torch.cat((next_feat, global_feats), dim= -1)
        return global_local_feats
    
    def getProbability(self, global_local_feats: torch.Tensor, mask_data: torch.Tensor):
        actor_out = self.actor_out2(F.relu(self.actor_out1(global_local_feats)))
        actor_out = actor_out.reshape(global_local_feats.shape[0], global_local_feats.shape[1]) * mask_data

        # Apply softmax only to non-zero elements of each row
        softmax_matrix = torch.zeros_like(actor_out)
        for i in range(actor_out.size(0)):
            row = actor_out[i]
            mask = (row != 0).float()
            softmax_row = torch.nn.functional.softmax(row[mask.bool()], dim=0)
            softmax_matrix[i, mask.bool()] = softmax_row

        return softmax_matrix
    
    def getValue(self, global_local_feats: torch.Tensor):
        critic_out = self.critic_out2(F.relu(self.critic_out1(global_local_feats)))
        return torch.mean(critic_out, dim=1)
    
    def getLastValue(self, robot_data: torch.Tensor, next_point_data: torch.Tensor, graph_data: torch.Tensor):
        global_local_feats = self.forward(robot_data, next_point_data, graph_data)
        value = self.getValue(global_local_feats)
        
        return value
    
    def getActionForTest(self, robot_data: torch.Tensor, next_point_data: torch.Tensor, 
                        graph_data: torch.Tensor, mask_data: torch.Tensor):
        global_local_feats = self.forward(robot_data, next_point_data, graph_data)
        
        probs = self.getProbability(global_local_feats, mask_data)
        
        return probs.argmax(dim = 1)
    
    def getAction(self, robot_data: torch.Tensor, next_point_data: torch.Tensor, 
                graph_data: torch.Tensor, mask_data: torch.Tensor):
        global_local_feats = self.forward(robot_data, next_point_data, graph_data)
        
        probs = self.getProbability(global_local_feats, mask_data)
        value = self.getValue(global_local_feats)
        
        dist = Categorical(probs)
        
        action = dist.sample()
        
        return action, value, dist.log_prob(action), dist.entropy()
    
    def evaluateAction(self, robot_data: torch.Tensor, next_point_data: torch.Tensor, 
                        graph_data: torch.Tensor, mask_data: torch.Tensor, action: torch.Tensor):
        global_local_feats = self.forward(robot_data, next_point_data, graph_data)
        probs = self.getProbability(global_local_feats, mask_data)
        value = self.getValue(global_local_feats)
        
        dist = Categorical(probs)
        
        return value, dist.log_prob(action), dist.entropy()

if __name__ == '__main__':
    robot_data = torch.randn(100, 5)
    next_point_data = torch.randn(100, 4, 4)
    graph_data = torch.randn(100, 5, 5, 8)
    mask_data = torch.randint(0, 2, (100, 4))
    net = PlanningNetwork()
    net.getActionForTest(robot_data, next_point_data, graph_data, mask_data)

