import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from g4satbench.models.mlp import MLP
from g4satbench.models.ln_lstm_cell import LayerNormBasicLSTMCell
from torch_scatter import scatter_sum, scatter_mean


class NeuroSAT(nn.Module):
    def __init__(self, opts):
        super(NeuroSAT, self).__init__()
        self.opts = opts
        self.l2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c_update = LayerNormBasicLSTMCell(self.opts.dim, self.opts.dim)
        self.l_update = LayerNormBasicLSTMCell(self.opts.dim * 2, self.opts.dim)
    
    def forward(self, l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb):
        # literal node state
        l_state = torch.zeros(l_size, self.opts.dim).to(self.opts.device)
        # clause node state
        c_state = torch.zeros(c_size, self.opts.dim).to(self.opts.device)

        l_embs = [l_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            # aggregate literal to clause messages
            l_msg_feat = self.l2c_msg_func(l_emb)
            l2c_msg = l_msg_feat[l_edge_index]
            l2c_msg_aggr = scatter_sum(l2c_msg, c_edge_index, dim=0, dim_size=c_size)
            
            # aggregate clause/negated literal to literal messages
            c_msg_feat = self.c2l_msg_func(c_emb)
            c2l_msg = c_msg_feat[c_edge_index]
            c2l_msg_aggr = scatter_sum(c2l_msg, l_edge_index, dim=0, dim_size=l_size)
            
            pl_emb, ul_emb = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
            l2l_msg = torch.cat([ul_emb, pl_emb], dim=1).reshape(l_size, -1)

            # update clause embeddings
            c_emb, c_state = self.c_update(l2c_msg_aggr, (c_emb, c_state))
            c_embs.append(c_emb)

            # update literal embeddings
            l_emb, l_state = self.l_update(torch.cat([c2l_msg_aggr, l2l_msg], dim=1), (l_emb, l_state))
            l_embs.append(l_emb)

        return l_embs, c_embs


class GGNN_LCG(nn.Module):
    def __init__(self, opts):
        super(GGNN_LCG, self).__init__()
        self.opts = opts
        self.l2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c_update = nn.GRUCell(self.opts.dim, self.opts.dim)
        self.l_update = nn.GRUCell(self.opts.dim * 2, self.opts.dim)
    
    def forward(self, l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb):
        l_embs = [l_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            # aggregate literal to clause messages
            l_msg_feat = self.l2c_msg_func(l_emb)
            l2c_msg = l_msg_feat[l_edge_index]
            l2c_msg_aggr = scatter_sum(l2c_msg, c_edge_index, dim=0, dim_size=c_size)

            # aggregate clause/negated literal to literal messages
            c_msg_feat = self.c2l_msg_func(c_emb)
            c2l_msg = c_msg_feat[c_edge_index]
            c2l_msg_aggr = scatter_sum(c2l_msg, l_edge_index, dim=0, dim_size=l_size)

            pl_emb, ul_emb = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
            l2l_msg = torch.cat([ul_emb, pl_emb], dim=1).reshape(l_size, -1)
                        
            # update clause embeddings
            c_emb = self.c_update(input=l2c_msg_aggr, hx=c_emb)
            c_embs.append(c_emb)

            # update literal embeddings
            l_emb = self.l_update(input=torch.cat([c2l_msg_aggr, l2l_msg], dim=1), hx=l_emb)
            l_embs.append(l_emb)

        return l_embs, c_embs



class GCN_LCG(nn.Module):
    def __init__(self, opts):
        super(GCN_LCG, self).__init__()
        self.opts = opts
        self.l2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        
        self.c_update = nn.Linear(self.opts.dim * 2, self.opts.dim)
        self.l_update = nn.Linear(self.opts.dim * 3, self.opts.dim)
    
    def forward(self, l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb):
        l_one = torch.ones((l_edge_index.size(0), 1), device=self.opts.device)
        l_deg = scatter_sum(l_one, l_edge_index, dim=0, dim_size=l_size)
        c_one = torch.ones((c_edge_index.size(0), 1), device=self.opts.device)
        c_deg = scatter_sum(c_one, c_edge_index, dim=0, dim_size=c_size)
        degree_norm = l_deg[l_edge_index].pow(0.5) * c_deg[c_edge_index].pow(0.5)

        l_embs = [l_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            # aggregate literal to clause messages
            l_msg_feat = self.l2c_msg_func(l_emb)
            l2c_msg = l_msg_feat[l_edge_index]
            l2c_msg_aggr = scatter_sum(l2c_msg / degree_norm, c_edge_index, dim=0, dim_size=c_size)

            # aggregate clause/negated literal to literal messages
            c_msg_feat = self.c2l_msg_func(c_emb)
            c2l_msg = c_msg_feat[c_edge_index]
            c2l_msg_aggr = scatter_sum(c2l_msg / degree_norm, l_edge_index, dim=0, dim_size=l_size)
            
            pl_emb, ul_emb = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
            l2l_msg = torch.cat([ul_emb, pl_emb], dim=1).reshape(l_size, -1)
            
            # update clause embeddings
            c_emb = self.c_update(torch.cat([c_emb, l2c_msg_aggr], dim=1))
            c_embs.append(c_emb)

            # update literal embeddings
            l_emb = self.l_update(torch.cat([l_emb, c2l_msg_aggr, l2l_msg], dim=1))
            l_embs.append(l_emb)

        return l_embs, c_embs


class GIN_LCG(nn.Module):
    def __init__(self, opts):
        super(GIN_LCG, self).__init__()
        self.opts = opts
        self.l2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        
        self.c_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 2, self.opts.dim, self.opts.dim, self.opts.activation)
        self.l_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 3, self.opts.dim, self.opts.dim, self.opts.activation)
    
    def forward(self, l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb):
        l_embs = [l_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            # aggregate literal to clause messages
            l_msg_feat = self.l2c_msg_func(l_emb)
            l2c_msg = l_msg_feat[l_edge_index]
            l2c_msg_aggr = scatter_sum(l2c_msg, c_edge_index, dim=0, dim_size=c_size)
            
            # aggregate clause/negated literal to literal messages
            c_msg_feat = self.c2l_msg_func(c_emb)
            c2l_msg = c_msg_feat[c_edge_index]
            c2l_msg_aggr = scatter_sum(c2l_msg, l_edge_index, dim=0, dim_size=l_size)

            pl_emb, ul_emb = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
            l2l_msg = torch.cat([ul_emb, pl_emb], dim=1).reshape(l_size, -1)
            
            # update clause embeddings
            c_emb = self.c_update(torch.cat([c_emb, l2c_msg_aggr], dim=1))
            c_embs.append(c_emb)
            
            # update literal embeddings
            l_emb = self.l_update(torch.cat([l_emb, c2l_msg_aggr, l2l_msg], dim=1))
            l_embs.append(l_emb)

        return l_embs, c_embs


class GNN_LCG(nn.Module):
    def __init__(self, opts):
        super(GNN_LCG, self).__init__()
        self.opts = opts
        if self.opts.init_emb == 'learned':
            self.l_init = nn.Parameter(torch.randn(1, self.opts.dim) * math.sqrt(2 / self.opts.dim))
            self.c_init = nn.Parameter(torch.randn(1, self.opts.dim) * math.sqrt(2 / self.opts.dim))
        
        if self.opts.model == 'neurosat':
            self.gnn = NeuroSAT(self.opts)
        elif self.opts.model == 'ggnn':
            self.gnn = GGNN_LCG(self.opts)
        elif self.opts.model == 'gcn':
            self.gnn = GCN_LCG(self.opts)
        elif self.opts.model == 'gin':
            self.gnn = GIN_LCG(self.opts)
        
        if self.opts.task == 'satisfiability':
            self.g_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        else:
            # self.opts.task == 'assignment' or self.opts.task == 'core_variable'
            self.l_readout = MLP(self.opts.n_mlp_layers, self.opts.dim * 2, self.opts.dim, 1, self.opts.activation)
        
        if hasattr(self.opts, 'use_contrastive_learning'):
            # temperature parameter
            self.tau = 0.5
            # projection head
            self.proj = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
    
    def forward(self, data):
        batch_size = data.num_graphs
        l_size = data.l_size.sum().item()
        c_size = data.c_size.sum().item()
        l_edge_index = data.l_edge_index
        c_edge_index = data.c_edge_index

        if self.opts.init_emb == 'learned':
            l_emb = (self.l_init).repeat(l_size, 1)
            c_emb = (self.c_init).repeat(c_size, 1)
        else:
            # self.opts.init_emb == 'random'
            l_emb = torch.randn(l_size, self.opts.dim, device=self.opts.device) * math.sqrt(2 / self.opts.dim)
            c_emb = torch.randn(c_size, self.opts.dim, device=self.opts.device) * math.sqrt(2 / self.opts.dim)

        l_embs, c_embs = self.gnn(l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb)
        
        if self.opts.task == 'satisfiability':
            l_batch = data.l_batch
            g_emb = scatter_mean(l_embs[-1], l_batch, dim=0, dim_size=batch_size)
            
            if hasattr(self.opts, 'use_contrastive_learning'):
                # SimCLR
                g_emb = self.proj(g_emb)
                h = F.normalize(g_emb, dim=1)
                sim = torch.exp(torch.mm(h, h.t()) / self.tau)
                # remove the similarity measure between two same objects
                mask = (1 - torch.eye(batch_size, device=self.opts.device)) 
                sim = sim * mask
                return sim
            else:
                g_logit = self.g_readout(g_emb).reshape(-1)
                return torch.sigmoid(g_logit)

        elif self.opts.task == 'assignment':
            if not hasattr(self.opts, 'decoding') or self.opts.decoding == 'standard':
                v_logit = self.l_readout(l_embs[-1].reshape(-1, self.opts.dim * 2)).reshape(-1)
                return torch.sigmoid(v_logit)
            
            elif self.opts.decoding == '2-clustering':
                # perform 2 clustering
                v_assign1 = []
                v_assign2 = []
                idx = 0
                for l_num in data.l_size:
                    l_emb = l_embs[-1][idx:idx+l_num]
                    idx += l_num

                    _, centers = kmeans(X=l_emb, num_clusters=2, distance='euclidean', tqdm_flag=0, device=self.opts.device)
                    distance1 = l_emb - centers[[0], :]
                    distance1 = (distance1 * distance1).sum(dim=1)

                    distance2 = l_emb - centers[[1], :]
                    distance2 = (distance2 * distance2).sum(dim=1)

                    pl_distance2, nl_distance2 = torch.chunk(distance2.reshape(-1, 2), 2, 1)
                    neg_distance2 = torch.cat([nl_distance2, pl_distance2], dim=1).reshape(-1)

                    distance = (distance1 + neg_distance2).reshape(-1, 2)
                    assignment1 = torch.argmin(distance, dim=1)
                    assignment2 = 1 - assignment1

                    v_assign1.append(assignment1)
                    v_assign2.append(assignment2)

                return [torch.cat(v_assign1, dim=0), torch.cat(v_assign2, dim=0)]
            
            else:
                assert self.opts.decoding == 'multiple_assignments'
                v_assigns = []
                for l_emb in l_embs:
                    v_logit = self.l_readout(l_emb.reshape(-1, self.opts.dim * 2)).reshape(-1)
                    v_assigns.append(torch.sigmoid(v_logit))
                return v_assigns
        
        else:
            assert self.opts.task == 'core_variable'
            v_logit = self.l_readout(l_embs[-1].reshape(-1, self.opts.dim * 2)).reshape(-1)
            return torch.sigmoid(v_logit)


class GGNN_VCG(nn.Module):
    def __init__(self, opts):
        super(GGNN_VCG, self).__init__()
        self.opts = opts
        self.p_v2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.n_v2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.p_c2v_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.n_c2v_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        
        self.c_update = nn.GRUCell(self.opts.dim * 2, self.opts.dim)
        self.v_update = nn.GRUCell(self.opts.dim * 2, self.opts.dim)
    
    def forward(self, v_size, c_size, v_edge_index, c_edge_index, p_edge_index, n_edge_index, v_emb, c_emb):
        v_embs = [v_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            # aggregate variable to clause messages
            p_v2c_msg_feat = self.p_v2c_msg_func(v_emb)
            p_v2c_msg = p_v2c_msg_feat[v_edge_index[p_edge_index]]
            p_v2c_msg_aggr = scatter_sum(p_v2c_msg, c_edge_index[p_edge_index], dim=0, dim_size=c_size)
            n_v2c_msg_feat = self.n_v2c_msg_func(v_emb)
            n_v2c_msg = n_v2c_msg_feat[v_edge_index[n_edge_index]]
            n_v2c_msg_aggr = scatter_sum(n_v2c_msg, c_edge_index[n_edge_index], dim=0, dim_size=c_size)

            # aggregate clause to variable messages
            p_c2v_msg_feat = self.p_c2v_msg_func(c_emb)
            p_c2v_msg = p_c2v_msg_feat[c_edge_index[p_edge_index]]
            p_c2v_msg_aggr = scatter_sum(p_c2v_msg, v_edge_index[p_edge_index], dim=0, dim_size=v_size)
            n_c2v_msg_feat = self.n_c2v_msg_func(c_emb)
            n_c2v_msg = n_c2v_msg_feat[c_edge_index[n_edge_index]]
            n_c2v_msg_aggr = scatter_sum(n_c2v_msg, v_edge_index[n_edge_index], dim=0, dim_size=v_size)

            # update clause embeddings
            c_emb = self.c_update(torch.cat([p_v2c_msg_aggr, n_v2c_msg_aggr], dim=1), c_emb)
            c_embs.append(c_emb)

            # update variable embeddings
            v_emb = self.v_update(torch.cat([p_c2v_msg_aggr, n_c2v_msg_aggr], dim=1), v_emb)
            v_embs.append(v_emb)

        return v_embs, c_embs



class GCN_VCG(nn.Module):
    def __init__(self, opts):
        super(GCN_VCG, self).__init__()
        self.opts = opts
        self.p_v2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.n_v2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.p_c2v_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.n_c2v_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        
        self.c_update = nn.Linear(self.opts.dim * 3, self.opts.dim)
        self.v_update = nn.Linear(self.opts.dim * 3, self.opts.dim)

    def forward(self, v_size, c_size, v_edge_index, c_edge_index, p_edge_index, n_edge_index, v_emb, c_emb):
        v_embs = [v_emb]
        c_embs = [c_emb]

        # calculate the degree of each clause/variable node
        p_v_one = torch.ones((v_edge_index[p_edge_index].size(0), 1), device=self.opts.device)
        p_v_deg = scatter_sum(p_v_one, v_edge_index[p_edge_index], dim=0, dim_size=v_size)
        p_v_deg[p_v_deg < 1] = 1
        n_v_one = torch.ones((v_edge_index[n_edge_index].size(0), 1), device=self.opts.device)
        n_v_deg = scatter_sum(n_v_one, v_edge_index[n_edge_index], dim=0, dim_size=v_size)
        n_v_deg[n_v_deg < 1] = 1

        p_c_one = torch.ones((c_edge_index[p_edge_index].size(0), 1), device=self.opts.device)
        p_c_deg = scatter_sum(p_c_one, c_edge_index[p_edge_index], dim=0, dim_size=c_size)
        p_c_deg[p_c_deg < 1] = 1
        n_c_one = torch.ones((c_edge_index[n_edge_index].size(0), 1), device=self.opts.device)
        n_c_deg = scatter_sum(n_c_one, c_edge_index[n_edge_index], dim=0, dim_size=c_size)
        n_c_deg[n_c_deg < 1] = 1

        p_norm = p_v_deg[v_edge_index[p_edge_index]].pow(0.5) * p_c_deg[c_edge_index[p_edge_index]].pow(0.5)
        n_norm = n_v_deg[v_edge_index[n_edge_index]].pow(0.5) * n_c_deg[c_edge_index[n_edge_index]].pow(0.5)

        for i in range(self.opts.n_iterations):
            # aggregate variable to clause messages
            p_v2c_msg_feat = self.p_v2c_msg_func(v_emb)
            p_v2c_msg = p_v2c_msg_feat[v_edge_index[p_edge_index]]
            p_v2c_msg_aggr = scatter_sum(p_v2c_msg / p_norm, c_edge_index[p_edge_index], dim=0, dim_size=c_size)
            n_v2c_msg_feat = self.n_v2c_msg_func(v_emb)
            n_v2c_msg = n_v2c_msg_feat[v_edge_index[n_edge_index]]
            n_v2c_msg_aggr = scatter_sum(n_v2c_msg / n_norm, c_edge_index[n_edge_index], dim=0, dim_size=c_size)

            # aggregate clause to variable messages
            p_c2v_msg_feat = self.p_c2v_msg_func(c_emb)
            p_c2v_msg = p_c2v_msg_feat[c_edge_index[p_edge_index]]
            p_c2v_msg_aggr = scatter_sum(p_c2v_msg / p_norm, v_edge_index[p_edge_index], dim=0, dim_size=v_size)
            n_c2v_msg_feat = self.n_c2v_msg_func(c_emb)
            n_c2v_msg = n_c2v_msg_feat[c_edge_index[n_edge_index]]
            n_c2v_msg_aggr = scatter_sum(n_c2v_msg / n_norm, v_edge_index[n_edge_index], dim=0, dim_size=v_size)

            # update clause embeddings
            c_emb = self.c_update(torch.cat([c_emb, p_v2c_msg_aggr, n_v2c_msg_aggr], dim=1))
            c_embs.append(c_emb)

            # update variable embeddings
            v_emb = self.v_update(torch.cat([v_emb, p_c2v_msg_aggr, n_c2v_msg_aggr], dim=1))
            v_embs.append(v_emb)

        return v_embs, c_embs


class GIN_VCG(nn.Module):
    def __init__(self, opts):
        super(GIN_VCG, self).__init__()
        self.opts = opts
        self.p_v2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.n_v2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.p_c2v_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.n_c2v_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        
        self.c_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 3, self.opts.dim, self.opts.dim, self.opts.activation)
        self.v_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 3, self.opts.dim, self.opts.dim, self.opts.activation)
    
    def forward(self, v_size, c_size, v_edge_index, c_edge_index, p_edge_index, n_edge_index, v_emb, c_emb):
        v_embs = [v_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            # aggregate variable to clause messages
            p_v2c_msg_feat = self.p_v2c_msg_func(v_emb)
            p_v2c_msg = p_v2c_msg_feat[v_edge_index[p_edge_index]]
            p_v2c_msg_aggr = scatter_sum(p_v2c_msg, c_edge_index[p_edge_index], dim=0, dim_size=c_size)
            n_v2c_msg_feat = self.n_v2c_msg_func(v_emb)
            n_v2c_msg = n_v2c_msg_feat[v_edge_index[n_edge_index]]
            n_v2c_msg_aggr = scatter_sum(n_v2c_msg, c_edge_index[n_edge_index], dim=0, dim_size=c_size)

            # aggregate clause to variable messages
            p_c2v_msg_feat = self.p_c2v_msg_func(c_emb)
            p_c2v_msg = p_c2v_msg_feat[c_edge_index[p_edge_index]]
            p_c2v_msg_aggr = scatter_sum(p_c2v_msg, v_edge_index[p_edge_index], dim=0, dim_size=v_size)
            n_c2v_msg_feat = self.n_c2v_msg_func(c_emb)
            n_c2v_msg = n_c2v_msg_feat[c_edge_index[n_edge_index]]
            n_c2v_msg_aggr = scatter_sum(n_c2v_msg, v_edge_index[n_edge_index], dim=0, dim_size=v_size)

            # update clause embeddings
            c_emb = self.c_update(torch.cat([c_emb, p_v2c_msg_aggr, n_v2c_msg_aggr], dim=1))
            c_embs.append(c_emb)

            # update variable embeddings
            v_emb = self.v_update(torch.cat([v_emb, p_c2v_msg_aggr, n_c2v_msg_aggr], dim=1))
            v_embs.append(v_emb)

        return v_embs, c_embs

class MS_ESFG(nn.Module):
    def __init__(self, opts):
        super(MS_ESFG, self).__init__()
        self.opts = opts
        self.dim = opts.dim
        self.n_mlp_layers = opts.n_mlp_layers
        self.activation = opts.activation

        # Aggregation functions for variables to clauses (positive and negative separately)
        self.aggl_pos = MLP(self.n_mlp_layers, self.dim, self.dim, self.dim, self.activation)
        self.aggl_neg = MLP(self.n_mlp_layers, self.dim, self.dim, self.dim, self.activation)

        # Aggregation functions for clauses to variables (positive and negative separately)
        self.aggc_pos = MLP(self.n_mlp_layers, self.dim, self.dim, self.dim, self.activation)
        self.aggc_neg = MLP(self.n_mlp_layers, self.dim, self.dim, self.dim, self.activation)

        # Clause and variable update modules
        self.clause_upd = nn.GRUCell(self.dim * 2, self.dim)
        self.variable_upd = nn.GRUCell(self.dim * 2, self.dim)

        # Readout layer to generate assignments
        self.readout = MLP(self.n_mlp_layers, self.dim, self.dim, 1, self.activation)

    def forward(self, v_size, c_size, v_edge_index, c_edge_index, p_edge_index, n_edge_index, v_emb, c_emb):
        # VCG-style message passing:
        # v_edge_index: [num_edges] - variable indices for each edge
        # c_edge_index: [num_edges] - clause indices for each edge
        # p_edge_index: indices of positive edges (in edge arrays)
        # n_edge_index: indices of negative edges (in edge arrays)
        v_embs, c_embs = [v_emb], [c_emb]

        for _ in range(self.opts.n_iterations):
            # Variable to clause messages (positive and negative separately)
            v_msg_pos = self.aggl_pos(v_emb)[v_edge_index[p_edge_index]]
            v_msg_neg = self.aggl_neg(v_emb)[v_edge_index[n_edge_index]]
            # Aggregate to clauses
            v2c_aggr_pos = scatter_sum(v_msg_pos, c_edge_index[p_edge_index], dim=0, dim_size=c_size)
            v2c_aggr_neg = scatter_sum(v_msg_neg, c_edge_index[n_edge_index], dim=0, dim_size=c_size)
            v2c_aggr = torch.cat([v2c_aggr_pos, v2c_aggr_neg], dim=1)

            # Clause to variable messages (positive and negative separately)
            c_msg_pos = self.aggc_pos(c_emb)[c_edge_index[p_edge_index]]
            c_msg_neg = self.aggc_neg(c_emb)[c_edge_index[n_edge_index]]
            # Aggregate to variables
            c2v_aggr_pos = scatter_sum(c_msg_pos, v_edge_index[p_edge_index], dim=0, dim_size=v_size)
            c2v_aggr_neg = scatter_sum(c_msg_neg, v_edge_index[n_edge_index], dim=0, dim_size=v_size)
            c2v_aggr = torch.cat([c2v_aggr_pos, c2v_aggr_neg], dim=1)

            # Clause update
            c_emb = self.clause_upd(v2c_aggr, c_emb)
            c_embs.append(c_emb)

            # Variable update
            v_emb = self.variable_upd(c2v_aggr, v_emb)
            v_embs.append(v_emb)

        return v_embs, c_embs


class GNN_VCG(nn.Module):
    def __init__(self, opts):
        super(GNN_VCG, self).__init__()
        self.opts = opts
        if self.opts.init_emb == 'learned':
            self.v_init = nn.Parameter(torch.randn(1, self.opts.dim) * math.sqrt(2 / self.opts.dim))
            self.c_init = nn.Parameter(torch.randn(1, self.opts.dim) * math.sqrt(2 / self.opts.dim))
        
        if self.opts.model == 'ggnn':
            self.gnn = GGNN_VCG(self.opts)
        elif self.opts.model == 'gcn':
            self.gnn = GCN_VCG(self.opts)
        elif self.opts.model == 'gin':
            self.gnn = GIN_VCG(self.opts)
        elif self.opts.model == 'ms_esfg':
            self.gnn = MS_ESFG(self.opts)
        
        if self.opts.task == 'satisfiability':
            self.g_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        else:
            # self.opts.task == 'assignment' or self.opts.task == 'core_variable'
            self.v_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)

        if hasattr(self.opts, 'use_contrastive_learning'):
            # temperature parameter
            self.tau = 0.5
            # projection head
            self.proj = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
    
    def forward(self, data):
        batch_size = data.num_graphs
        v_size = data.v_size.sum().item()
        c_size = data.c_size.sum().item()

        v_edge_index = data.v_edge_index
        c_edge_index = data.c_edge_index
        p_edge_index = data.p_edge_index
        n_edge_index = data.n_edge_index

        if self.opts.init_emb == 'learned':
            v_emb = (self.v_init).repeat(v_size, 1)
            c_emb = (self.c_init).repeat(c_size, 1)
        else:
            # self.opts.init_emb == 'random'
            v_emb = torch.randn(v_size, self.opts.dim, device=self.opts.device) * math.sqrt(2 / self.opts.dim)
            c_emb = torch.randn(c_size, self.opts.dim, device=self.opts.device) * math.sqrt(2 / self.opts.dim)

        v_embs, c_embs = self.gnn(v_size, c_size, v_edge_index, c_edge_index, p_edge_index, n_edge_index, v_emb, c_emb)
        
        if self.opts.task == 'satisfiability':
            v_batch = data.v_batch
            g_emb = scatter_mean(v_embs[-1], v_batch, dim=0, dim_size=batch_size)
            
            if hasattr(self.opts, 'use_contrastive_learning'):
                # SimCLR
                g_emb = self.proj(g_emb)
                h = F.normalize(g_emb, dim=1)
                sim = torch.exp(torch.mm(h, h.t()) / self.tau)
                # remove the similarity measure between two same objects
                mask = (1 - torch.eye(batch_size, device=self.opts.device))
                sim = sim * mask
                return sim
            else:
                g_logit = self.g_readout(g_emb).reshape(-1)
                return torch.sigmoid(g_logit)

        elif self.opts.task == 'assignment':
            # 2-clustering can not be applied to VCG
            if not hasattr(self.opts, 'decoding') or self.opts.decoding == 'standard':
                v_logit = self.v_readout(v_embs[-1]).reshape(-1)
                return torch.sigmoid(v_logit)
            else:
                assert self.opts.decoding == 'multiple_assignments'
                v_assigns = []
                for v_emb in v_embs:
                    v_logit = self.v_readout(v_emb).reshape(-1)
                    v_assigns.append(torch.sigmoid(v_logit))
                return v_assigns
        
        else:
            assert self.opts.task == 'core_variable'
            v_logit = self.v_readout(v_embs[-1]).reshape(-1)
            return torch.sigmoid(v_logit)


def GNN(opts):
    if opts.graph == 'lcg':
        return GNN_LCG(opts)
    else:
        # opts.graph == 'vcg'
        return GNN_VCG(opts)
