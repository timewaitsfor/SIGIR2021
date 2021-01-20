import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.inits import reset

try:
    from pykeops.torch import LazyTensor
except ImportError:
    LazyTensor = None

EPS = 1e-8


def masked_softmax(src, mask, dim=-1):
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    return out


def to_sparse(x, mask):
    return x[mask]


def to_dense(x, mask):
    out = x.new_zeros(tuple(mask.size()) + (x.size(-1), ))
    out[mask] = x
    return out

class DGMC(torch.nn.Module):
    r"""The *Deep Graph Matching Consensus* module which first matches nodes
    locally via a graph neural network :math:`\Psi_{\theta_1}`, and then
    updates correspondence scores iteratively by reaching for neighborhood
    consensus via a second graph neural network :math:`\Psi_{\theta_2}`.

    .. note::
        See the `PyTorch Geometric introductory tutorial
        <https://pytorch-geometric.readthedocs.io/en/latest/notes/
        introduction.html>`_ for a detailed overview of the used GNN modules
        and the respective data format.

    Args:
        psi_1 (torch.nn.Module): The first GNN :math:`\Psi_{\theta_1}` which
            takes in node features :obj:`x`, edge connectivity
            :obj:`edge_index`, and optional edge features :obj:`edge_attr` and
            computes node embeddings.
        psi_2 (torch.nn.Module): The second GNN :math:`\Psi_{\theta_2}` which
            takes in node features :obj:`x`, edge connectivity
            :obj:`edge_index`, and optional edge features :obj:`edge_attr` and
            validates for neighborhood consensus.
            :obj:`psi_2` needs to hold the attributes :obj:`in_channels` and
            :obj:`out_channels` which indicates the dimensionality of randomly
            drawn node indicator functions and the output dimensionality of
            :obj:`psi_2`, respectively.
        num_steps (int): Number of consensus iterations.
        k (int, optional): Sparsity parameter. If set to :obj:`-1`, will
            not sparsify initial correspondence rankings. (default: :obj:`-1`)
        detach (bool, optional): If set to :obj:`True`, will detach the
            computation of :math:`\Psi_{\theta_1}` from the current computation
            graph. (default: :obj:`False`)
    """
    def __init__(self, psi_1, psi_2, num_steps, k=-1, detach=False):
        super(DGMC, self).__init__()

        self.psi_1 = psi_1
        self.psi_2 = psi_2
        self.num_steps = num_steps
        self.k = k
        self.detach = detach
        self.backend = 'auto'

        self.rel_lin = Lin(self.psi_1.in_channels * 2, self.psi_1.in_channels)

        self.mlp = Seq(
            Lin(psi_2.out_channels, psi_2.out_channels),
            ReLU(),
            Lin(psi_2.out_channels, 1),
        )

    def reset_parameters(self):
        self.psi_1.reset_parameters()
        self.psi_2.reset_parameters()
        reset(self.mlp)

    def __top_k__(self, x_s, x_t):  # pragma: no cover
        r"""Memory-efficient top-k correspondence computation."""
        if LazyTensor is not None:
            x_s = x_s.unsqueeze(-2)  # [..., n_s, 1, d]
            x_t = x_t.unsqueeze(-3)  # [..., 1, n_t, d]
            x_s, x_t = LazyTensor(x_s), LazyTensor(x_t)
            S_ij = (-x_s * x_t).sum(dim=-1)
            return S_ij.argKmin(self.k, dim=2, backend=self.backend)
        else:
            x_s = x_s  # [..., n_s, d]
            x_t = x_t.transpose(-1, -2)  # [..., d, n_t]
            S_ij = x_s @ x_t
            return S_ij.topk(self.k, dim=2)[1]

    def __rel_top_k__(self, x_s, x_t, train_ry, test_ry):  # pragma: no cover
        r"""Memory-efficient top-k correspondence computation."""
        if LazyTensor is not None:
            x_s = x_s.unsqueeze(-2)  # [..., n_s, 1, d]
            x_t = x_t.unsqueeze(-3)  # [..., 1, n_t, d]
            x_s, x_t = LazyTensor(x_s), LazyTensor(x_t)
            S_ij = (-x_s * x_t).sum(dim=-1)
            return S_ij.argKmin(self.k, dim=2, backend=self.backend)
        else:
            x_s = x_s  # [..., n_s, d]
            x_t = x_t.transpose(-1, -2)  # [..., d, n_t]
            S_ij_temp = x_s @ x_t
            S_ij = torch.zeros_like(S_ij_temp)

            train_rtup_s = train_ry[0, :]
            train_rtup_t = train_ry[1, :]

            test_rtup_s = test_ry[0, :]
            test_rtup_t = test_ry[1, :]

            for i in range(S_ij.size(1)):
                if i in train_rtup_s:
                    S_ij[:, i, train_rtup_t] = S_ij_temp[:, i, train_rtup_t]
                elif i in test_rtup_s:
                    S_ij[:, i, test_rtup_t] = S_ij_temp[:, i, test_rtup_t]
                else:
                    S_ij[:, i, :] = S_ij_temp[:, i, :]

            return S_ij.topk(self.k, dim=2)[1]

    def __top_test_k__(self, x_s, x_t, test_y):  # pragma: no cover
        r"""Memory-efficient top-k correspondence computation."""

        test_x_s = x_s[:, test_y[0, :]]
        test_x_t = x_t[:, test_y[1, :]]

        if LazyTensor is not None:
            test_x_s = test_x_s.unsqueeze(-2)  # [..., n_s, 1, d]
            test_x_t = test_x_t.unsqueeze(-3)  # [..., 1, n_t, d]
            test_x_s, test_x_t = LazyTensor(test_x_s), LazyTensor(test_x_t)
            test_S_ij = (-test_x_s * test_x_t).sum(dim=-1)
            test_S = test_S_ij.argKmin(self.k, dim=2, backend=self.backend)

            x_s = x_s.unsqueeze(-2)  # [..., n_s, 1, d]
            x_t = x_t.unsqueeze(-3)  # [..., 1, n_t, d]
            x_s, x_t = LazyTensor(x_s), LazyTensor(x_t)
            S_ij = (-x_s * x_t).sum(dim=-1)
            S = S_ij.argKmin(self.k, dim=2, backend=self.backend)

            for i, s_yi in enumerate(test_y[0, :]):
                # t_yi = test_y[1, i]
                this_candidate = test_S[:, i]
                this_candidate_ids = test_y[1, this_candidate]
                S[:, s_yi] = this_candidate_ids

            return S
        else:
            x_s = x_s  # [..., n_s, d]
            x_t = x_t.transpose(-1, -2)  # [..., d, n_t]
            S_ij = x_s @ x_t
            return S_ij.topk(self.k, dim=2)[1]

    def __include_gt__(self, S_idx, s_mask, y):
        r"""Includes the ground-truth values in :obj:`y` to the index tensor
        :obj:`S_idx`."""
        (B, N_s), (row, col), k = s_mask.size(), y, S_idx.size(-1)

        gt_mask = (S_idx[s_mask][row] != col.view(-1, 1)).all(dim=-1)

        # print(torch.sum(gt_mask))

        sparse_mask = gt_mask.new_zeros((s_mask.sum(), ))
        sparse_mask[row] = gt_mask

        dense_mask = sparse_mask.new_zeros((B, N_s))
        dense_mask[s_mask] = sparse_mask
        last_entry = torch.zeros(k, dtype=torch.bool, device=gt_mask.device)
        last_entry[-1] = 1
        dense_mask = dense_mask.view(B, N_s, 1) * last_entry.view(1, 1, k)
        return S_idx.masked_scatter(dense_mask, col[gt_mask])

    def __rel_include_gt__(self, S_idx, s_mask, y):
        (B, N_s), (row, col), k = s_mask.size(), y, S_idx.size(-1)

        gt_mask = (S_idx[s_mask][row] != col.view(-1, 1)).all(dim=-1)


        for i, line in enumerate(gt_mask):
            if line == torch.tensor(1, dtype=torch.bool):
                this_gt = col[i]
                # print(S_idx[:,row[i],:])
                S_idx[:,row[i],-1] = this_gt
                # print(S_idx[:,row[i],:])

        return S_idx

    def forward(self, ex_s, rx_s, edge_index_s, edge_attr_s, batch_s, ex_t, rx_t,
                edge_index_t, edge_attr_t, batch_t, train_y=None, train_ry=None, test_y=None, test_ry=None):

        if rx_s.size(1) == 2*ex_s.size(1):
            rx_s = self.rel_lin(rx_s)
            rx_t = self.rel_lin(rx_t)

        x_s = torch.cat((ex_s, rx_s), dim=0)
        x_t = torch.cat((ex_t, rx_t), dim=0)

        e_s_num = ex_s.size(0)
        e_t_num = ex_t.size(0)

        r_s_num = rx_s.size(0)
        r_t_num = rx_t.size(0)

        h_s = self.psi_1(x_s, e_s_num, r_s_num, edge_index_s)
        h_t = self.psi_1(x_t, e_t_num, r_t_num, edge_index_t)

        h_s, h_t = (h_s.detach(), h_t.detach()) if self.detach else (h_s, h_t)

        eh_s = h_s[:ex_s.size(0), :]
        eh_t = h_t[:ex_t.size(0), :]

        rh_s = h_s[ex_s.size(0):, :]
        rh_t = h_t[ex_t.size(0):, :]

        eh_s, es_mask = to_dense_batch(eh_s, batch_s, fill_value=0)
        rh_s, rs_mask = to_dense_batch(rh_s, batch_s, fill_value=0)

        eh_t, et_mask = to_dense_batch(eh_t, batch_t, fill_value=0)
        rh_t, rt_mask = to_dense_batch(rh_t, batch_t, fill_value=0)

        h_s, s_mask = to_dense_batch(h_s, batch_s, fill_value=0)
        h_t, t_mask = to_dense_batch(h_t, batch_t, fill_value=0)

        # assert h_s.size(0) == h_t.size(0), 'Encountered unequal batch-sizes'

        eB, eN_s, eC_out, eN_t = eh_s.size(0), ex_s.size(0), eh_s.size(-1), ex_t.size(0)
        rB, rN_s, rC_out, rN_t = eh_s.size(0), rx_s.size(0), eh_s.size(-1), rx_t.size(0)
        (B, N_s, C_out), N_t = h_s.size(), h_t.size(1)

        R_in, R_out = self.psi_2.in_channels, self.psi_2.out_channels

        if self.k < 1:
            # ------ Dense variant ------ #
            # S_hat = h_s @ h_t.transpose(-1, -2)  # [B, N_s, N_t, C_out]
            # S_mask = s_mask.view(B, N_s, 1) & t_mask.view(B, 1, N_t)
            # S_0 = masked_softmax(S_hat, S_mask, dim=-1)[s_mask]
            #
            # for _ in range(self.num_steps):
            #     S = masked_softmax(S_hat, S_mask, dim=-1)
            #     r_s = torch.randn((B, N_s, R_in), dtype=h_s.dtype,
            #                       device=h_s.device)
            #     r_t = S.transpose(-1, -2) @ r_s
            #
            #     r_s, r_t = to_sparse(r_s, s_mask), to_sparse(r_t, t_mask)
            #     o_s = self.psi_2(r_s, edge_index_s, edge_attr_s)
            #     o_t = self.psi_2(r_t, edge_index_t, edge_attr_t)
            #     o_s, o_t = to_dense(o_s, s_mask), to_dense(o_t, t_mask)
            #
            #     D = o_s.view(B, N_s, 1, R_out) - o_t.view(B, 1, N_t, R_out)
            #     S_hat = S_hat + self.mlp(D).squeeze(-1).masked_fill(~S_mask, 0)
            #
            # S_L = masked_softmax(S_hat, S_mask, dim=-1)[s_mask]
            #
            # return S_0, S_L
            pass
        else:
            # ------ Sparse variant ------ #

            # In addition to the top-k, randomly sample negative examples and
            # ensure that the ground-truth is included as a sparse entry.
            if self.training and train_y is not None:
                eS_idx = self.__top_k__(eh_s, eh_t)  # [B, N_s, k]
                rS_idx = self.__top_k__(rh_s, rh_t)  # [B, N_s, k]


                e_rnd_size = (eB, eN_s, min(self.k, eN_t - self.k))
                eS_rnd_idx = torch.randint(eN_t, e_rnd_size, dtype=torch.long,
                                          device=eS_idx.device)
                eS_idx = torch.cat([eS_idx, eS_rnd_idx], dim=-1)
                eS_idx = self.__include_gt__(eS_idx, es_mask, train_y)

                r_rnd_size = (rB, rN_s, min(self.k, rN_t - self.k))
                rS_rnd_idx = torch.randint(rN_t, r_rnd_size, dtype=torch.long,
                                           device=rS_idx.device)
                rS_idx = torch.cat([rS_idx, rS_rnd_idx], dim=-1)
                rS_idx = self.__rel_include_gt__(rS_idx, rs_mask, train_ry)
            else:
                eS_idx = self.__top_test_k__(eh_s, eh_t, test_y)  # [B, N_s, k]
                rS_idx = self.__top_test_k__(rh_s, rh_t, test_ry)  # [B, N_s, k]

            # entity and relation shared
            k = eS_idx.size(-1)

            # entity S
            tmp_es = eh_s.view(eB, eN_s, 1, eC_out)
            eidx = eS_idx.view(eB, eN_s * k, 1).expand(-1, -1, eC_out)
            tmp_et = torch.gather(eh_t.view(eB, eN_t, eC_out), -2, eidx)
            eS_hat = (tmp_es * tmp_et.view(eB, eN_s, k, eC_out)).sum(dim=-1)
            eS_0 = eS_hat.softmax(dim=-1)[es_mask]

            # relation S
            tmp_rs = rh_s.view(rB, rN_s, 1, rC_out)
            ridx = rS_idx.view(rB, rN_s * k, 1).expand(-1, -1, rC_out)
            tmp_rt = torch.gather(rh_t.view(rB, rN_t, rC_out), -2, ridx)
            rS_hat = (tmp_rs * tmp_rt.view(rB, rN_s, k, rC_out)).sum(dim=-1)
            rS_0 = rS_hat.softmax(dim=-1)[rs_mask]

            for _ in range(self.num_steps):
                eS = eS_hat.softmax(dim=-1)
                rS = rS_hat.softmax(dim=-1)

                '''
                entity 
                '''
                er_s = torch.randn((eB, eN_s, R_in), dtype=eh_s.dtype,
                                  device=eh_s.device)
                tmp_et = er_s.view(eB, eN_s, 1, R_in) * eS.view(eB, eN_s, k, 1)
                tmp_et = tmp_et.view(eB, eN_s * k, R_in)

                # eS_idx = torch.cat((eS_idx, rS_idx), dim=1)
                eidx = eS_idx.view(eB, eN_s * k, 1)
                er_t = scatter_add(tmp_et, eidx, dim=1, dim_size=eN_t)
                er_s, er_t = to_sparse(er_s, es_mask), to_sparse(er_t, et_mask)


                '''
                relation 
                '''
                rr_s = torch.randn((rB, rN_s, R_in), dtype=rh_s.dtype,
                                   device=rh_s.device)
                tmp_rt = rr_s.view(rB, rN_s, 1, R_in) * rS.view(rB, rN_s, k, 1)
                tmp_rt = tmp_rt.view(rB, rN_s * k, R_in)

                # eS_idx = torch.cat((eS_idx, rS_idx), dim=1)
                ridx = rS_idx.view(rB, rN_s * k, 1)
                rr_t = scatter_add(tmp_rt, ridx, dim=1, dim_size=rN_t)
                rr_s, rr_t = to_sparse(rr_s, rs_mask), to_sparse(rr_t, rt_mask)

                r_s = torch.cat((er_s, rr_s), dim=0)
                r_t = torch.cat((er_t, rr_t), dim=0)

                o_s = self.psi_2(r_s, e_s_num, r_s_num, edge_index_s)
                o_t = self.psi_2(r_t, e_t_num, r_t_num, edge_index_t)

                o_s, o_t = to_dense(o_s, s_mask), to_dense(o_t, t_mask)
                o_s = o_s.view(B, N_s, 1, R_out).expand(-1, -1, k, -1)

                eo_s = o_s[:, :ex_s.size(0), :, :]
                eo_t = o_t[:, :ex_t.size(0), :]

                ro_s = o_s[:, ex_s.size(0):, :, :]
                ro_t = o_t[:, ex_t.size(0):, :]

                eidx = eS_idx.view(eB, eN_s * k, 1).expand(-1, -1, R_out)
                tmp_et = torch.gather(eo_t.view(eB, eN_t, R_out), -2, eidx)

                ridx = rS_idx.view(rB, rN_s * k, 1).expand(-1, -1, R_out)
                tmp_rt = torch.gather(ro_t.view(rB, rN_t, R_out), -2, ridx)

                eD = eo_s - tmp_et.view(eB, eN_s, k, R_out)
                rD = ro_s - tmp_rt.view(rB, rN_s, k, R_out)

                eS_hat = eS_hat + self.mlp(eD).squeeze(-1)
                rS_hat = rS_hat + self.mlp(rD).squeeze(-1)

            # entity ...
            eS_L = eS_hat.softmax(dim=-1)[es_mask]
            eS_idx = eS_idx[es_mask]

            # Convert sparse layout to `torch.sparse_coo_tensor`.
            erow = torch.arange(ex_s.size(0), device=eS_idx.device)
            erow = erow.view(-1, 1).repeat(1, k)
            eidx = torch.stack([erow.view(-1), eS_idx.view(-1)], dim=0)
            esize = torch.Size([ex_s.size(0), eN_t])

            eS_sparse_0 = torch.sparse_coo_tensor(
                eidx, eS_0.view(-1), esize, requires_grad=eS_0.requires_grad)
            eS_sparse_0.__idx__ = eS_idx
            eS_sparse_0.__val__ = eS_0

            eS_sparse_L = torch.sparse_coo_tensor(
                eidx, eS_L.view(-1), esize, requires_grad=eS_L.requires_grad)
            eS_sparse_L.__idx__ = eS_idx
            eS_sparse_L.__val__ = eS_L

            # relation ...
            rS_L = rS_hat.softmax(dim=-1)[rs_mask]
            rS_idx = rS_idx[rs_mask]

            # Convert sparse layout to `torch.sparse_coo_tensor`.
            rrow = torch.arange(rx_s.size(0), device=rS_idx.device)
            rrow = rrow.view(-1, 1).repeat(1, k)
            ridx = torch.stack([rrow.view(-1), rS_idx.view(-1)], dim=0)
            rsize = torch.Size([rx_s.size(0), rN_t])

            rS_sparse_0 = torch.sparse_coo_tensor(
                ridx, rS_0.view(-1), rsize, requires_grad=rS_0.requires_grad)
            rS_sparse_0.__idx__ = rS_idx
            rS_sparse_0.__val__ = rS_0

            rS_sparse_L = torch.sparse_coo_tensor(
                ridx, rS_L.view(-1), rsize, requires_grad=rS_L.requires_grad)
            rS_sparse_L.__idx__ = rS_idx
            rS_sparse_L.__val__ = rS_L

            return eS_sparse_0, eS_sparse_L, rS_sparse_0, rS_sparse_L

    def loss(self, eS, rS, ey, ry, reduction='mean'):
        assert reduction in ['none', 'mean', 'sum']
        if not eS.is_sparse:
            val = eS[ey[0], ey[1]]
        else:
            assert eS.__idx__ is not None and eS.__val__ is not None
            emask = eS.__idx__[ey[0]] == ey[1].view(-1, 1)
            eval = eS.__val__[[ey[0]]][emask]

            rmask = rS.__idx__[ry[0]] == ry[1].view(-1, 1)
            rval = rS.__val__[[ry[0]]][rmask]

        enll = -torch.log(eval + EPS)
        e_loss = enll if reduction == 'none' else getattr(torch, reduction)(enll)

        rnll = -torch.log(rval + EPS)
        r_loss = rnll if reduction == 'none' else getattr(torch, reduction)(rnll)

        loss = e_loss + r_loss

        return loss

    def acc(self, eS, rS, ey, ry, reduction='mean'):
        r"""Computes the accuracy of correspondence predictions.

        Args:
            S (Tensor): Sparse or dense correspondence matrix of shape
                :obj:`[batch_size * num_nodes, num_nodes]`.
            y (LongTensor): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]`.
            reduction (string, optional): Specifies the reduction to apply to
                the output: :obj:`'mean'|'sum'`. (default: :obj:`'mean'`)
        """
        assert reduction in ['mean', 'sum']
        dgmc_e1_to_e2_scores = {}
        if not eS.is_sparse:
            epred = eS[ey[0]].argmax(dim=-1)
        else:
            assert eS.__idx__ is not None and eS.__val__ is not None
            epred = eS.__idx__[ey[0], eS.__val__[ey[0]].argmax(dim=-1)]

            # for ry_id in ry[0]:
            #     for i, (target_id, target_val) in enumerate(zip(rS.__idx__[ry_id, :], rS.__val__[ry_id, :])):
            #         source_list = edge_dict1[ry_id.item()]
            #         target_list = edge_dict2[target_id.item()]
            #
            #         p_num = len(list(set(source_list).intersection(set(target_list))))
            #         ht_num = len(set(source_list + target_list))
            #         if ht_num == 0:
            #             continue
            #
            #         p_score = (20*p_num)/ht_num + 1
            #         rS.__idx__[ry_id, i] = rS.__idx__[ry_id, i]*p_score

            rpred = rS.__idx__[ry[0], rS.__val__[ry[0]].argmax(dim=-1)]
            # rpred = rS.__idx__[ry[0], rS.__val__[ry[0]].argsort(dim=-1)]

            for i, e1 in enumerate(ey[0,:]):
                dgmc_e1_to_e2_scores[e1.item()] =[]
                for j, e2 in enumerate(eS.__idx__[e1]):
                    if e2 == ey[1,i]:
                        label = 1
                    else:
                        label = 0
                    dgmc_e1_to_e2_scores[e1.item()].append((e2.item(), eS.__val__[i, j].item(), label))

        ecorrect = (epred == ey[1]).sum().item()
        rcorrect = (rpred == ry[1]).sum().item()

        e_res = ecorrect / ey.size(1) if reduction == 'mean' else ecorrect
        r_res = rcorrect / ry.size(1) if reduction == 'mean' else ecorrect
        return e_res, r_res, dgmc_e1_to_e2_scores

    def hits_at_k(self, k, eS, rS, ey, ry, reduction='mean'):
        assert reduction in ['mean', 'sum']
        if not eS.is_sparse:
            epred = eS[ey[0]].argsort(dim=-1, descending=True)[:, :k]
        else:
            assert eS.__idx__ is not None and eS.__val__ is not None
            eperm = eS.__val__[ey[0]].argsort(dim=-1, descending=True)[:, :k]
            epred = torch.gather(eS.__idx__[ey[0]], -1, eperm)

            rperm = rS.__val__[ry[0]].argsort(dim=-1, descending=True)[:, :k]
            rpred = torch.gather(rS.__idx__[ry[0]], -1, rperm)

        ecorrect = (epred == ey[1].view(-1, 1)).sum().item()
        rcorrect = (rpred == ry[1].view(-1, 1)).sum().item()

        e_res = ecorrect / ey.size(1) if reduction == 'mean' else ecorrect
        r_res = rcorrect / ry.size(1) if reduction == 'mean' else rcorrect

        return e_res, r_res

    # def mrr(self, eS, ey, reduction='mean'):
    #     assert reduction in ['mean', 'sum']
    #     k =100
    #     dgmc_e1_to_e2_scores = {}
    #     if not eS.is_sparse:
    #         epred = eS[ey[0]].argsort(dim=-1, descending=True)[:, :k]
    #     else:
    #         assert eS.__idx__ is not None and eS.__val__ is not None
    #         eperm = eS.__val__[ey[0]].argsort(dim=-1, descending=True)[:, :k]
    #         epred = torch.gather(eS.__idx__[ey[0]], -1, eperm)
    #
    #     ecorrect = (epred == ey[1].view(-1, 1)).sum().item()
    #     e_mrr = ecorrect / ey.size(1) if reduction == 'mean' else ecorrect
    #
    # return e_mrr

    def __repr__(self):
        return ('{}(\n'
                '    psi_1={},\n'
                '    psi_2={},\n'
                '    num_steps={}, k={}\n)').format(self.__class__.__name__,
                                                    self.psi_1, self.psi_2,
                                                    self.num_steps, self.k)
