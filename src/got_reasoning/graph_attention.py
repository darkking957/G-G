# src/got_reasoning/graph_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
import math


class GraphAttentionLayer(nn.Module):
    """单个图注意力层"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.6, 
                 alpha: float = 0.2, concat: bool = True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # 可学习的权重矩阵
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # 注意力机制参数
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # LeakyReLU激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: 节点特征矩阵 [N, in_features]
            adj: 邻接矩阵 [N, N]
        
        Returns:
            新的节点特征 [N, out_features]
        """
        N = h.size(0)
        
        # 线性变换
        h_transformed = torch.mm(h, self.W)  # [N, out_features]
        
        # 计算注意力系数
        a_input = self._prepare_attentional_mechanism_input(h_transformed)  # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N, N]
        
        # 掩码处理（只保留邻接节点的注意力）
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Softmax归一化
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 应用注意力权重
        h_prime = torch.matmul(attention, h_transformed)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
            
    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        """准备注意力机制的输入"""
        N = Wh.size(0)
        
        # 使用广播机制高效计算
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # [N*N, out_features]
        Wh_repeated_alternating = Wh.repeat(N, 1)  # [N*N, out_features]
        
        # 拼接特征
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # [N*N, 2*out_features]
        
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class MultiHeadGraphAttention(nn.Module):
    """多头图注意力层"""
    
    def __init__(self, n_heads: int, in_features: int, out_features: int, 
                 dropout: float = 0.6, alpha: float = 0.2):
        super(MultiHeadGraphAttention, self).__init__()
        self.n_heads = n_heads
        
        # 创建多个注意力头
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(n_heads)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """多头注意力前向传播"""
        # 对每个头计算注意力
        head_outputs = [att(h, adj) for att in self.attentions]
        
        # 拼接所有头的输出
        h = torch.cat(head_outputs, dim=1)
        h = self.dropout(h)
        
        return h


class GraphAttentionNetwork(nn.Module):
    """完整的图注意力网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_heads: int = 8, dropout: float = 0.6, alpha: float = 0.2):
        super(GraphAttentionNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # 第一层：多头注意力
        self.attention1 = MultiHeadGraphAttention(
            n_heads=num_heads,
            in_features=input_dim,
            out_features=hidden_dim,
            dropout=dropout,
            alpha=alpha
        )
        
        # 第二层：单头注意力（输出层）
        self.attention2 = GraphAttentionLayer(
            in_features=hidden_dim * num_heads,
            out_features=output_dim,
            dropout=dropout,
            alpha=alpha,
            concat=False
        )
        
        # 可选的额外层
        self.use_residual = True
        if self.use_residual and input_dim == output_dim:
            self.residual = nn.Linear(input_dim, output_dim)
        else:
            self.residual = None
            
        # 归一化层
        self.layer_norm1 = nn.LayerNorm(hidden_dim * num_heads)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [N, input_dim]
            adj: 邻接矩阵 [N, N]
            
        Returns:
            输出特征 [N, output_dim]
        """
        # 第一层多头注意力
        h = self.attention1(x, adj)
        h = self.layer_norm1(h)
        h = F.elu(h)
        
        # 第二层注意力
        h = self.attention2(h, adj)
        
        # 残差连接（如果可用）
        if self.residual is not None:
            h = h + self.residual(x)
            
        h = self.layer_norm2(h)
        
        return h
        
    def get_attention_weights(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """获取注意力权重用于可视化"""
        # 这里简化实现，只返回第一层第一个头的注意力权重
        with torch.no_grad():
            N = x.size(0)
            h_transformed = torch.mm(x, self.attention1.attentions[0].W)
            a_input = self.attention1.attentions[0]._prepare_attentional_mechanism_input(h_transformed)
            e = self.attention1.attentions[0].leakyrelu(torch.matmul(a_input, self.attention1.attentions[0].a).squeeze(2))
            
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            
        return attention


class ThoughtGraphAttention(nn.Module):
    """专门用于思想图的注意力网络"""
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256, 
                 num_heads: int = 8, num_layers: int = 2):
        super(ThoughtGraphAttention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 思想类型嵌入
        self.type_embeddings = nn.Embedding(10, 64)  # 最多10种思想类型
        self.type_to_id = {
            "question": 0,
            "path": 1,
            "aggregated": 2,
            "refined": 3,
            "merged": 4,
            "bridge": 5,
            "validation": 6,
            "feedback": 7
        }
        
        # 投影层
        self.input_projection = nn.Linear(embedding_dim + 64, hidden_dim)
        
        # 多层GAT
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(
                    MultiHeadGraphAttention(num_heads, hidden_dim, hidden_dim // num_heads)
                )
            else:
                self.gat_layers.append(
                    MultiHeadGraphAttention(num_heads, hidden_dim, hidden_dim // num_heads)
                )
                
        # 输出层
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)
        
        # 分数预测头
        self.score_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features: torch.Tensor, node_types: torch.Tensor, 
                adj_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: 节点特征 [N, embedding_dim]
            node_types: 节点类型ID [N]
            adj_matrix: 邻接矩阵 [N, N]
            
        Returns:
            (增强的节点特征, 预测的节点分数)
        """
        # 获取类型嵌入
        type_embeds = self.type_embeddings(node_types)  # [N, 64]
        
        # 拼接特征和类型嵌入
        h = torch.cat([node_features, type_embeds], dim=-1)  # [N, embedding_dim + 64]
        
        # 输入投影
        h = self.input_projection(h)  # [N, hidden_dim]
        h = F.relu(h)
        
        # 多层GAT
        for i, gat_layer in enumerate(self.gat_layers):
            h_new = gat_layer(h, adj_matrix)
            if i > 0:  # 残差连接
                h = h + h_new
            else:
                h = h_new
                
        # 输出投影
        h_out = self.output_projection(h)  # [N, embedding_dim]
        
        # 预测分数
        scores = self.score_head(h_out).squeeze(-1)  # [N]
        
        return h_out, scores
        
    def encode_thought_type(self, thought_type: str) -> int:
        """将思想类型转换为ID"""
        return self.type_to_id.get(thought_type, 0)


def create_thought_adjacency_matrix(thought_graph, device='cuda'):
    """
    为思想图创建邻接矩阵
    
    Args:
        thought_graph: ThoughtGraph实例
        device: 计算设备
        
    Returns:
        (邻接矩阵, 节点ID列表)
    """
    node_ids = list(thought_graph.nodes.keys())
    n = len(node_ids)
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    
    # 创建邻接矩阵
    adj = torch.zeros((n, n), device=device)
    
    for edge in thought_graph.edges:
        if edge.source in node_id_to_idx and edge.target in node_id_to_idx:
            i = node_id_to_idx[edge.source]
            j = node_id_to_idx[edge.target]
            adj[i, j] = edge.weight
            
    # 添加自连接
    adj = adj + torch.eye(n, device=device)
    
    return adj, node_ids


def apply_gat_to_thought_graph(thought_graph, gat_model, node_embeddings, device='cuda'):
    """
    将GAT应用到思想图
    
    Args:
        thought_graph: ThoughtGraph实例
        gat_model: GraphAttentionNetwork或ThoughtGraphAttention实例
        node_embeddings: 节点嵌入字典 {node_id: embedding}
        device: 计算设备
        
    Returns:
        增强的节点嵌入字典
    """
    # 创建邻接矩阵
    adj, node_ids = create_thought_adjacency_matrix(thought_graph, device)
    
    # 准备特征矩阵
    features = []
    node_types = []
    
    for node_id in node_ids:
        node = thought_graph.nodes[node_id]
        
        # 获取嵌入
        if node_id in node_embeddings:
            embedding = node_embeddings[node_id]
        else:
            # 使用随机初始化
            embedding = torch.randn(gat_model.embedding_dim, device=device)
            
        features.append(embedding)
        
        # 获取类型ID（如果使用ThoughtGraphAttention）
        if isinstance(gat_model, ThoughtGraphAttention):
            type_id = gat_model.encode_thought_type(node.node_type)
            node_types.append(type_id)
            
    features = torch.stack(features)
    
    # 应用GAT
    if isinstance(gat_model, ThoughtGraphAttention):
        node_types = torch.tensor(node_types, dtype=torch.long, device=device)
        enhanced_features, scores = gat_model(features, node_types, adj)
        
        # 更新节点分数
        for i, node_id in enumerate(node_ids):
            thought_graph.nodes[node_id].score = scores[i].item()
    else:
        enhanced_features = gat_model(features, adj)
        
    # 创建增强的嵌入字典
    enhanced_embeddings = {}
    for i, node_id in enumerate(node_ids):
        enhanced_embeddings[node_id] = enhanced_features[i]
        
    return enhanced_embeddings