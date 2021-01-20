from utils import *
from param import *

from torch.nn import Parameter
from torch.nn.init import xavier_normal_

import torch
import inspect

from torch_scatter import scatter
import numpy as np

def entlist2emb(Model,entids,entid2data,cuda_num):
    """
    return basic bert unit output embedding of entities
    """
    batch_token_ids = []
    batch_mask_ids = []
    for eid in entids:
        temp_token_ids = entid2data[eid][0]
        temp_mask_ids = entid2data[eid][1]

        batch_token_ids.append(temp_token_ids)
        batch_mask_ids.append(temp_mask_ids)

    batch_token_ids = torch.LongTensor(batch_token_ids).cuda(cuda_num)
    batch_mask_ids = torch.FloatTensor(batch_mask_ids).cuda(cuda_num)

    batch_emb = Model(batch_token_ids,batch_mask_ids)
    del batch_token_ids
    del batch_mask_ids
    return batch_emb

def get_param(shape):
	param = Parameter(torch.Tensor(*shape))
	xavier_normal_(param.data)
	return param

def scatter_(name, src, index, dim_size=None):
	r"""Aggregates all values from the :attr:`src` tensor at the indices
	specified in the :attr:`index` tensor along the first dimension.
	If multiple indices reference the same location, their contributions
	are aggregated according to :attr:`name` (either :obj:`"add"`,
	:obj:`"mean"` or :obj:`"max"`).

	Args:
		name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
			:obj:`"max"`).
		src (Tensor): The source tensor.
		index (LongTensor): The indices of elements to scatter.
		dim_size (int, optional): Automatically create output tensor with size
			:attr:`dim_size` in the first dimension. If set to :attr:`None`, a
			minimal sized output tensor is returned. (default: :obj:`None`)

	:rtype: :class:`Tensor`
	"""
	if name == 'add': name = 'sum'
	assert name in ['sum', 'mean', 'max']
	out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=name)
	return out[0] if isinstance(out, tuple) else out

class MessagePassing(torch.nn.Module):
	r"""Base class for creating message passing layers

	.. math::
		\mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
		\square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
		\left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

	where :math:`\square` denotes a differentiable, permutation invariant
	function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
	and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
	MLPs.
	See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
	create_gnn.html>`__ for the accompanying tutorial.

	"""

	def __init__(self, aggr='add'):
		super(MessagePassing, self).__init__()

		self.message_args = inspect.getargspec(self.message)[0][1:]	# In the defined message function: get the list of arguments as list of string| For eg. in rgcn this will be ['x_j', 'edge_type', 'edge_norm'] (arguments of message function)
		self.update_args  = inspect.getargspec(self.update)[0][2:]	# Same for update function starting from 3rd argument | first=self, second=out

	def propagate(self, edge_index, **kwargs):
		r"""The initial call to start propagating messages.
		Takes in an aggregation scheme (:obj:`"add"`, :obj:`"mean"` or
		:obj:`"max"`), the edge indices, and all additional data which is
		needed to construct messages and to update node embeddings."""

		kwargs['edge_index'] = edge_index

		size = None
		message_args = []
		for arg in self.message_args:
			if arg[-2:] == '_i':					# If arguments ends with _i then include indic
				tmp  = kwargs[arg[:-2]]				# Take the front part of the variable | Mostly it will be 'x',
				size = tmp.size(0)
				message_args.append(tmp[edge_index[0]])		# Lookup for head entities in edges
			elif arg[-2:] == '_j':
				tmp  = kwargs[arg[:-2]]				# tmp = kwargs['x']
				size = tmp.size(0)
				message_args.append(tmp[edge_index[1]])		# Lookup for tail entities in edges
			else:
				message_args.append(kwargs[arg])		# Take things from kwargs

		update_args = [kwargs[arg] for arg in self.update_args]		# Take update args from kwargs

		out = self.message(*message_args)
		out = self.update(out, *update_args)

		return out

	def message(self, x_j):  # pragma: no cover
		r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
		for each edge in :math:`(i,j) \in \mathcal{E}`.
		Can take any argument which was initially passed to :meth:`propagate`.
		In addition, features can be lifted to the source node :math:`i` and
		target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
		variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

		return x_j

	def update(self, aggr_out):  # pragma: no cover
		r"""Updates node embeddings in analogy to
		:math:`\gamma_{\mathbf{\Theta}}` for each node
		:math:`i \in \mathcal{V}`.
		Takes in the output of aggregation as first argument and any argument
		which was initially passed to :meth:`propagate`."""

		return aggr_out

def get_tradic_edges(edge_index, edge_type, entity_cnt):
	edge_index = np.array(edge_index).T
	tradic_edge_index = []
	for i in range(edge_index.shape[1]):
		head = edge_index[0, i]
		tail = edge_index[1, i]
		relation = edge_type[i] + entity_cnt
		tradic_edge_index.append([head, relation])
		tradic_edge_index.append([head, tail])

	for i in range(edge_index.shape[1]):
		head = edge_index[0, i]
		tail = edge_index[1, i]
		relation = edge_type[i] + entity_cnt
		tradic_edge_index.append([tail, head])
		tradic_edge_index.append([tail, relation])
		# edge_type.append() 在忧郁要不要加上表示是关系还是实体的token

	for i in range(edge_index.shape[1]):
		head = edge_index[0, i]
		tail = edge_index[1, i]
		relation = edge_type[i] + entity_cnt
		tradic_edge_index.append([relation, tail])

	for i in range(edge_index.shape[1]):
		head = edge_index[0, i]
		tail = edge_index[1, i]
		relation = edge_type[i] + entity_cnt
		# relation = edge_type[i]
		tradic_edge_index.append([relation, head])

	return tradic_edge_index

def get_mrr(e1_to_e2andscores, test_topk=k):
	all_test_num = len(e1_to_e2andscores.keys())  # test set size.
	result_labels = []
	for e, value_list in e1_to_e2andscores.items():
		v_list = value_list
		v_list.sort(key=lambda x: x[1], reverse=True)
		label_list = [label for e2, score, label in v_list]
		label_list = label_list[:test_topk]
		result_labels.append(label_list)
	result_labels = np.array(result_labels)
	result_labels = result_labels.sum(axis=0).tolist()
	topk_list = []
	mrr_rank = []
	for i in range(test_topk):
		# rank_index = np.where(result_labels == i)[0][0]
		this_rank = 1 / (i + 1)
		mrr_rank += [this_rank] * result_labels[i]
		nums = sum(result_labels[:i + 1])
		topk_list.append(round(nums / all_test_num, 5))

	return float(np.sum(mrr_rank)/all_test_num)

