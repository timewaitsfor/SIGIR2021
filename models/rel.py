import torch
from torch.nn import Linear as Lin, BatchNorm1d as BN
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from helper import *

class RelConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(RelConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin1 = Lin(in_channels, out_channels, bias=False)
        self.lin2 = Lin(in_channels, out_channels, bias=False)
        self.root = Lin(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.root.reset_parameters()

    def forward(self, x, edge_index):
        """"""
        self.flow = 'source_to_target'
        out1 = self.propagate(edge_index, x=self.lin1(x))
        self.flow = 'target_to_source'
        out2 = self.propagate(edge_index, x=self.lin2(x))
        return self.root(x) + out1 + out2

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class RelCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, batch_norm=False,
                 cat=True, lin=True, dropout=0.0):
        super(RelCNN, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.cat = cat
        self.lin = lin
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RelConv(in_channels, out_channels))
            self.batch_norms.append(BN(out_channels))
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = Lin(in_channels, out_channels)
        else:
            self.out_channels = in_channels

        self.reset_parameters()

    def reset_parameters(self):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            conv.reset_parameters()
            batch_norm.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, *args):
        """"""
        xs = [x]

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(xs[-1], edge_index)
            x = batch_norm(F.relu(x)) if self.batch_norm else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ('{}({}, {}, num_layers={}, batch_norm={}, cat={}, lin={}, '
                'dropout={})').format(self.__class__.__name__,
                                      self.in_channels, self.out_channels,
                                      self.num_layers, self.batch_norm,
                                      self.cat, self.lin, self.dropout)


class CompGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, act=lambda x: x):
        super(self.__class__, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.device = None

        self.dropout = 0.1

        self.w_loop_e = get_param((in_channels, out_channels))
        self.w_loop_r = get_param((in_channels, out_channels))
        self.w_ipt_e = get_param((in_channels, out_channels))
        self.w_ipt_r = get_param((in_channels, out_channels))
        self.w_opt_e = get_param((in_channels, out_channels))
        self.w_opt_r = get_param((in_channels, out_channels))

        self.ent_proximity_linear = Lin(in_channels * 3, out_channels)
        self.rel_proximity_linear = Lin(in_channels * 3, out_channels)

        self.drop = torch.nn.Dropout(self.dropout)
        self.bn = torch.nn.BatchNorm1d(out_channels)

        self.bias = False

        if self.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))


    def forward(self, x, num_ent, num_rel, edge_index):
        if self.device is None:
            self.device = edge_index.device

        # rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1)
        # num_ent = x.size(0)

        num_nodes = num_ent + num_rel

        x_dtype = x.dtype

        opt_e_edges = int(num_edges/3)
        ipt_e_edges = int(num_edges*2/3)

        opt_r_edges = int(num_edges*5/6)
        # ipt_r_edges = num_edges * 2 / 3

        self.opt_e_index, self.ipt_e_index = edge_index[:, :opt_e_edges], edge_index[:, opt_e_edges:ipt_e_edges]
        self.opt_r_index, self.ipt_r_index = edge_index[:, ipt_e_edges:opt_r_edges], edge_index[:, opt_r_edges:]

        self.loop_e_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_r_index = torch.stack([torch.arange(num_ent, num_ent+num_rel), torch.arange(num_ent, num_ent+num_rel)]).to(self.device)

        self.opt_e_norm = self.compute_ent_norm(self.opt_e_index, num_ent, num_rel, x_dtype)
        self.ipt_e_norm = self.compute_ent_norm(self.ipt_e_index, num_ent, num_rel, x_dtype)

        self.opt_r_norm = self.compute_rel_norm(self.opt_r_index, num_rel, num_ent, x_dtype)
        self.ipt_r_norm = self.compute_rel_norm(self.ipt_r_index, num_rel, num_ent, x_dtype)

        aggr = 'add'

        opt_e_res = self.propagate(self.opt_e_index, x=x, edge_norm=self.opt_e_norm, mode='opt_e')
        loop_e_res = self.propagate(self.loop_e_index, x=x, edge_norm=None, mode='loop_e')
        ipt_e_res = self.propagate(self.ipt_e_index, x=x, edge_norm=self.ipt_e_norm, mode='ipt_e')

        opt_e_res = scatter_(aggr, opt_e_res, self.opt_e_index[0], dim_size=num_nodes)  # Aggregated neighbors for each vertex
        loop_e_res = scatter_(aggr, loop_e_res, self.loop_e_index[0], dim_size=num_nodes)  # Aggregated neighbors for each vertex
        ipt_e_res = scatter_(aggr, ipt_e_res, self.ipt_e_index[0], dim_size=num_nodes)  # Aggregated neighbors for each vertex

        opt_r_res = self.propagate(self.opt_r_index, x=x, edge_norm=self.opt_r_norm, mode='opt_r')
        loop_r_res = self.propagate(self.loop_r_index, x=x, edge_norm=None, mode='loop_r')
        ipt_r_res = self.propagate(self.ipt_r_index, x=x, edge_norm=self.ipt_r_norm, mode='ipt_r')

        # comp_r_res = torch.cat((opt_r_res, ipt_r_res), dim=-1)
        # comp_r_res = opt_r_res*ipt_r_res
        # comp_r_res = scatter_(aggr, comp_r_res, self.opt_r_index[0], dim_size=num_nodes)  # Aggregated neighbors for each vertex
        opt_r_res = scatter_(aggr, opt_r_res, self.opt_r_index[0], dim_size=num_nodes)  # Aggregated neighbors for each vertex
        loop_r_res = scatter_(aggr, loop_r_res, self.loop_r_index[0], dim_size=num_nodes)  # Aggregated neighbors for each vertex
        ipt_r_res = scatter_(aggr, ipt_r_res, self.ipt_r_index[0], dim_size=num_nodes)  # Aggregated neighbors for each vertex


        # e_out = self.drop(opt_e_res) * (1 / 3) + self.drop(loop_e_res) * (1 / 3) + self.drop(ipt_e_res) * (1 / 3)
        # r_out = self.drop(opt_r_res) * (1 / 3) + self.drop(loop_r_res) * (1 / 3) + self.drop(ipt_r_res) * (1 / 3)

        e_out = torch.cat((self.drop(opt_e_res), self.drop(loop_e_res), self.drop(ipt_e_res)), dim=-1)
        r_out = torch.cat((self.drop(opt_r_res), self.drop(loop_r_res), self.drop(ipt_r_res)), dim=-1)
        # r_out = torch.cat((self.drop(comp_r_res), self.drop(loop_r_res)), dim=-1)

        e_out = self.ent_proximity_linear(e_out)
        r_out = self.rel_proximity_linear(r_out)

        return self.act(e_out[:num_ent]), self.act(r_out[num_ent:])

    def compute_ent_norm(self, edge_index, num_ent, num_rel, x_dtype):
        row, col = edge_index
        total_num = num_ent + num_rel

        edge_weight = torch.ones_like(row).float()
        deg = degree(row, num_ent, dtype=x_dtype)  # Summing number of weights of the edges
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == torch.tensor(float('inf'))] = 0

        total_deg = degree(col, total_num, dtype=x_dtype)
        total_deg_inv = total_deg.pow(-0.5)  # D^{-0.5}
        total_deg_inv[total_deg_inv == torch.tensor(float('inf'))] = 0

        norm = deg_inv[row] * edge_weight * total_deg_inv[col]  # D^{-0.5}
        return norm

    def compute_rel_norm(self, edge_index, num_ent, num_rel, x_dtype):
        row, col = edge_index
        total_num = num_ent + num_rel

        edge_weight = torch.ones_like(row).float()
        deg = degree(row, total_num, dtype=x_dtype)  # Summing number of weights of the edges
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == torch.tensor(float('inf'))] = 0

        total_deg = degree(col, total_num, dtype=x_dtype)
        total_deg_inv = total_deg.pow(-0.5)  # D^{-0.5}
        total_deg_inv[total_deg_inv == torch.tensor(float('inf'))] = 0

        norm = deg_inv[row] * edge_weight * total_deg_inv[col]  # D^{-0.5}
        return norm

    def message(self, x_j, edge_norm, mode):
        weight = getattr(self, 'w_{}'.format(mode))
        out = torch.mm(x_j, weight)
        return out if edge_norm is None else out * edge_norm.view(-1, 1)

class CompGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, batch_norm=False,
                 cat=True, lin=True, dropout=0.0):
        super(CompGNN, self).__init__()

        self.act = torch.tanh
        # self.act = F.relu
        # self.conv1 = CompGCNConv(in_channels, out_channels, act=self.act)
        # self.conv2 = CompGCNConv(out_channels, out_channels, act=self.act)

        self.in_channels = in_channels
        self.dropout = dropout
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.cat = cat
        self.lin = lin

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(CompGCNConv(in_channels, out_channels, act=self.act))
            self.batch_norms.append(BN(out_channels))
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = Lin(in_channels, out_channels)
        else:
            self.out_channels = in_channels

    def forward(self, x, num_ent, num_rel, edge_index, *args):
        """"""
        xs = [x]

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            ex, rx = conv(xs[-1], num_ent, num_rel, edge_index)
            erx = torch.cat((ex, rx), dim=0)
            x = batch_norm(self.act(erx)) if self.batch_norm else self.act(erx)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = self.final(x) if self.lin else x

        return x