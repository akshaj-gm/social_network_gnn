{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "456ed869-89ab-4610-bbf8-c0d89d503c20",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Looking in links: https://pytorch-geometric.com/whl/torch-2.0.1%2Bcu118.html\n",
      "Requirement already satisfied: torch-scatter in /Users/akshaj.g/mamba/lib/python3.10/site-packages (2.1.1)\n",
      "Looking in links: https://pytorch-geometric.com/whl/torch-2.0.1%2Bcu118.html\n",
      "Requirement already satisfied: torch-sparse in /Users/akshaj.g/mamba/lib/python3.10/site-packages (0.6.17)\n",
      "Requirement already satisfied: scipy in /Users/akshaj.g/mamba/lib/python3.10/site-packages (from torch-sparse) (1.10.0)\n",
      "Requirement already satisfied: numpy<1.27.0,>=1.19.5 in /Users/akshaj.g/mamba/lib/python3.10/site-packages (from scipy->torch-sparse) (1.24.1)\n",
      "Requirement already satisfied: torch-geometric in /Users/akshaj.g/mamba/lib/python3.10/site-packages (2.3.1)\n",
      "Requirement already satisfied: tqdm in /Users/akshaj.g/mamba/lib/python3.10/site-packages (from torch-geometric) (4.64.1)\n",
      "Requirement already satisfied: numpy in /Users/akshaj.g/mamba/lib/python3.10/site-packages (from torch-geometric) (1.24.1)\n",
      "Requirement already satisfied: scipy in /Users/akshaj.g/mamba/lib/python3.10/site-packages (from torch-geometric) (1.10.0)\n",
      "Requirement already satisfied: jinja2 in /Users/akshaj.g/mamba/lib/python3.10/site-packages (from torch-geometric) (3.1.2)\n",
      "Requirement already satisfied: requests in /Users/akshaj.g/mamba/lib/python3.10/site-packages (from torch-geometric) (2.28.1)\n",
      "Requirement already satisfied: pyparsing in /Users/akshaj.g/mamba/lib/python3.10/site-packages (from torch-geometric) (3.0.9)\n",
      "Requirement already satisfied: scikit-learn in /Users/akshaj.g/mamba/lib/python3.10/site-packages (from torch-geometric) (1.2.0)\n",
      "Requirement already satisfied: psutil>=5.8.0 in /Users/akshaj.g/mamba/lib/python3.10/site-packages (from torch-geometric) (5.9.4)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in /Users/akshaj.g/mamba/lib/python3.10/site-packages (from jinja2->torch-geometric) (2.1.1)\n",
      "Requirement already satisfied: charset-normalizer<3,>=2 in /Users/akshaj.g/mamba/lib/python3.10/site-packages (from requests->torch-geometric) (2.1.1)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /Users/akshaj.g/mamba/lib/python3.10/site-packages (from requests->torch-geometric) (3.4)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in /Users/akshaj.g/mamba/lib/python3.10/site-packages (from requests->torch-geometric) (1.26.11)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /Users/akshaj.g/mamba/lib/python3.10/site-packages (from requests->torch-geometric) (2022.12.7)\n",
      "Requirement already satisfied: joblib>=1.1.1 in /Users/akshaj.g/mamba/lib/python3.10/site-packages (from scikit-learn->torch-geometric) (1.2.0)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in /Users/akshaj.g/mamba/lib/python3.10/site-packages (from scikit-learn->torch-geometric) (3.1.0)\n"
     ]
    }
   ],
   "source": [
    "# Install torch geometric\n",
    "import os\n",
    "if 'IS_GRADESCOPE_ENV' not in os.environ:\n",
    "  !pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.1%2Bcu118.html\n",
    "  !pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.1%2Bcu118.html\n",
    "  !pip install torch-geometric"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ac5110e1-9a6d-4276-82ee-88b20e5e06c2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pip in /Users/akshaj.g/mamba/lib/python3.10/site-packages (23.2.1)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install --upgrade pip"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "a7124823-9dc4-4764-8769-9bc6d4aec5d0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'2.3.1'"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import torch_geometric\n",
    "torch_geometric.__version__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "66339363-231b-497f-afd5-b193fd0557ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np \n",
    "import pandas as pd \n",
    "import matplotlib.pyplot as plt\n",
    "import torch\n",
    "from collections import defaultdict\n",
    "from typing import Any, Dict, Iterable, List, Optional, Tuple, Union\n",
    "\n",
    "import scipy.sparse\n",
    "from torch import Tensor\n",
    "from torch.utils.dlpack import from_dlpack, to_dlpack\n",
    "\n",
    "from torch_geometric.utils.num_nodes import maybe_num_nodes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "146dd7cd-5493-4efd-9149-5771d1000f77",
   "metadata": {},
   "outputs": [],
   "source": [
    "train=pd.read_csv('/Users/akshaj.g/Desktop/ml/social network /FacebookRecruiting/train.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "ed28148c-1471-4e51-9bbc-db800cbff9ff",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1862220\n"
     ]
    }
   ],
   "source": [
    "edge_index = torch.tensor(train.values).T\n",
    "num_nodes = edge_index.max().item()\n",
    "print(num_nodes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ca8b5d7f-6b9f-44a1-89bb-f5fd7ef0088e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([2, 9437519])"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "edge_index.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a54e3785-c79a-4178-9032-2866473278d2",
   "metadata": {},
   "outputs": [],
   "source": [
    "def to_scipy_sparse_matrix(\n",
    "    edge_index: Tensor,\n",
    "    edge_attr: Optional[Tensor] = None,\n",
    "    num_nodes: Optional[int] = None,\n",
    ") -> scipy.sparse.coo_matrix:\n",
    "    row, col = edge_index.cpu()\n",
    "\n",
    "    if edge_attr is None:\n",
    "        edge_attr = torch.ones(row.size(0))\n",
    "    else:\n",
    "        edge_attr = edge_attr.view(-1).cpu()\n",
    "        assert edge_attr.size(0) == row.size(0)\n",
    "\n",
    "    N = maybe_num_nodes(edge_index, num_nodes)\n",
    "    out = scipy.sparse.coo_matrix(\n",
    "        (edge_attr.numpy(), (row.numpy(), col.numpy())), (N, N))\n",
    "    return out\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a91a32f8-7cda-4f48-80a4-21396933002e",
   "metadata": {},
   "outputs": [],
   "source": [
    "adj_mat=to_scipy_sparse_matrix(edge_index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "3402570b-1e98-43b5-ba11-ea1b3ba765fa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[33mWARNING: Skipping /opt/homebrew/lib/python3.11/site-packages/numpy-1.26.0-py3.11.egg-info due to invalid metadata entry 'name'\u001b[0m\u001b[33m\n",
      "\u001b[0mRequirement already satisfied: tensordict in /opt/homebrew/lib/python3.11/site-packages (0.1.2)\n",
      "Requirement already satisfied: torch in /opt/homebrew/lib/python3.11/site-packages (from tensordict) (2.0.1)\n",
      "Requirement already satisfied: numpy in /opt/homebrew/lib/python3.11/site-packages (from tensordict) (1.25.2)\n",
      "Requirement already satisfied: cloudpickle in /opt/homebrew/lib/python3.11/site-packages (from tensordict) (2.2.1)\n",
      "Requirement already satisfied: filelock in /opt/homebrew/lib/python3.11/site-packages (from torch->tensordict) (3.12.3)\n",
      "Requirement already satisfied: typing-extensions in /opt/homebrew/lib/python3.11/site-packages (from torch->tensordict) (4.7.1)\n",
      "Requirement already satisfied: sympy in /opt/homebrew/lib/python3.11/site-packages (from torch->tensordict) (1.12)\n",
      "Requirement already satisfied: networkx in /opt/homebrew/lib/python3.11/site-packages (from torch->tensordict) (2.8.8)\n",
      "Requirement already satisfied: jinja2 in /opt/homebrew/lib/python3.11/site-packages (from torch->tensordict) (3.1.2)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in /opt/homebrew/lib/python3.11/site-packages (from jinja2->torch->tensordict) (2.1.3)\n",
      "Requirement already satisfied: mpmath>=0.19 in /opt/homebrew/lib/python3.11/site-packages (from sympy->torch->tensordict) (1.3.0)\n",
      "\u001b[33mWARNING: Skipping /opt/homebrew/lib/python3.11/site-packages/numpy-1.26.0-py3.11.egg-info due to invalid metadata entry 'name'\u001b[0m\u001b[33m\n",
      "\u001b[0m\u001b[33mWARNING: Skipping /opt/homebrew/lib/python3.11/site-packages/numpy-1.26.0-py3.11.egg-info due to invalid metadata entry 'name'\u001b[0m\u001b[33m\n",
      "\u001b[0m\u001b[33mWARNING: Skipping /opt/homebrew/lib/python3.11/site-packages/numpy-1.26.0-py3.11.egg-info due to invalid metadata entry 'name'\u001b[0m\u001b[33m\n",
      "\u001b[0m"
     ]
    }
   ],
   "source": [
    "!pip3 install tensordict\n",
    "import torch.nn as nn\n",
    "from tensordict import TensorDict\n",
    "from tensordict.nn import TensorDictModule, TensorDictSequential"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "491e11e2-b6cd-46f5-bf4f-d318da78eaef",
   "metadata": {},
   "outputs": [],
   "source": [
    "node_features=defaultdict(torch.Tensor)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "2b493308-f5b9-44a6-b242-c8b353a77ede",
   "metadata": {},
   "outputs": [],
   "source": [
    "def node_degree(edge_list, adj_mat, num_nodes):\n",
    "  degree=torch.tensor(adj_mat.sum(axis=1).A1, dtype=torch.long)\n",
    "  degree = torch.where(degree > 0, degree, torch.zeros_like(degree))\n",
    "  if degree.shape[0] < num_nodes:\n",
    "    degree = torch.cat((degree, torch.zeros(num_nodes - degree.shape[0], dtype=torch.long)))\n",
    "\n",
    "  return degree\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "ca0804ea-287b-453c-9a5b-2a4a42022004",
   "metadata": {},
   "outputs": [],
   "source": [
    "degree_t=node_degree(edge_index,adj_mat,num_nodes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "4d46c961-9332-46d1-9a5c-86f5fb57b2a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch_scatter\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "\n",
    "import torch_geometric.nn as pyg_nn\n",
    "import torch_geometric.utils as pyg_utils\n",
    "\n",
    "from torch import Tensor\n",
    "from typing import Union, Tuple, Optional\n",
    "from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,\n",
    "                                    OptTensor)\n",
    "\n",
    "from torch.nn import Parameter, Linear\n",
    "from torch_sparse import SparseTensor, set_diag\n",
    "from torch_geometric.nn.conv import MessagePassing\n",
    "from torch_geometric.utils import remove_self_loops, add_self_loops, softmax\n",
    "\n",
    "class GNNStack(torch.nn.Module):\n",
    "    def __init__(self, input_dim, hidden_dim, output_dim, args, emb=False):\n",
    "        super(GNNStack, self).__init__()\n",
    "        conv_model = self.build_conv_model(args.model_type)\n",
    "        self.convs = nn.ModuleList()\n",
    "        self.convs.append(conv_model(input_dim, hidden_dim))\n",
    "        assert (args.num_layers >= 1), 'Number of layers is not >=1'\n",
    "        for l in range(args.num_layers-1):\n",
    "            self.convs.append(conv_model(args.heads * hidden_dim, hidden_dim))\n",
    "\n",
    "\n",
    "        self.post_mp = nn.Sequential(\n",
    "            nn.Linear(args.heads * hidden_dim, hidden_dim), nn.Dropout(args.dropout),\n",
    "            nn.Linear(hidden_dim, output_dim))\n",
    "\n",
    "        self.dropout = args.dropout\n",
    "        self.num_layers = args.num_layers\n",
    "\n",
    "        self.emb = emb\n",
    "\n",
    "    def build_conv_model(self, model_type):\n",
    "        if model_type == 'GraphSage':\n",
    "            return GraphSage\n",
    "        elif model_type == 'GAT':\n",
    "            return GAT\n",
    "        \n",
    "    def forward(self, x, pos_edge_index, neg_edge_index):\n",
    "        for i in range(self.num_layers):\n",
    "            x = self.convs[i](x, pos_edge_index, neg_edge_index)\n",
    "            x = F.relu(x)\n",
    "            x = F.dropout(x, p=self.dropout, training=self.training)\n",
    "\n",
    "        x = self.post_mp(x)\n",
    "\n",
    "        if self.emb:\n",
    "            return x\n",
    "\n",
    "        return F.log_softmax(x, dim=1)\n",
    "    \n",
    "    def get_node_embeddings(self, x):\n",
    "        for i in range(self.num_layers):\n",
    "            x = self.convs[i](x, edge_index)\n",
    "            x = F.relu(x)\n",
    "            x = F.dropout(x, p=self.dropout, training=False)  # Set training to False to get deterministic output\n",
    "        return x \n",
    "    \n",
    "    def loss(self, pred, label):\n",
    "        return F.nll_loss(pred, label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "491c23e3-71f6-4f55-b4a8-475078dbeb72",
   "metadata": {},
   "outputs": [],
   "source": [
    "class GraphSage(MessagePassing):\n",
    "    \n",
    "    def __init__(self, in_channels, out_channels, normalize = True,\n",
    "                 bias = False, **kwargs):  \n",
    "        super(GraphSage, self).__init__(**kwargs)\n",
    "\n",
    "        self.in_channels = in_channels\n",
    "        self.out_channels = out_channels\n",
    "        self.normalize = normalize\n",
    "\n",
    "        self.lin_l = None\n",
    "        self.lin_r = None\n",
    "        self.lin_l = nn.Linear(self.in_channels, self.out_channels)\n",
    "        self.lin_r = nn.Linear(self.in_channels, self.out_channels)\n",
    "        self.reset_parameters()\n",
    "\n",
    "    def reset_parameters(self):\n",
    "        self.lin_l.reset_parameters()\n",
    "        self.lin_r.reset_parameters()\n",
    "\n",
    "    '''def forward(self, x, pos_edge_index, neg_edge_index, size=None):\n",
    "        # Propagate messages for positive edges\n",
    "        x_j_pos = self.propagate(pos_edge_index, x=x)\n",
    "        \n",
    "        # Propagate messages for negative edges\n",
    "        x_j_neg = self.propagate(neg_edge_index, x=x)\n",
    "        \n",
    "        # Create a mask tensor for missing nodes in pos_edge_index\n",
    "        mask_pos = torch.zeros(x.size(0), device=x.device)\n",
    "        mask_pos[pos_edge_index[0]] = 1.0\n",
    "        \n",
    "        # Create a mask tensor for missing nodes in neg_edge_index\n",
    "        mask_neg = torch.zeros(x.size(0), device=x.device)\n",
    "        mask_neg[neg_edge_index[0]] = 1.0\n",
    "        \n",
    "        # Apply the masks to embeddings\n",
    "        x_j_pos = x_j_pos * mask_pos.unsqueeze(1)\n",
    "        x_j_neg = x_j_neg * mask_neg.unsqueeze(1)\n",
    "        \n",
    "        # Perform the linear transformations\n",
    "        out = self.lin_l(x) + self.lin_r(x_j_pos) - self.lin_r(x_j_neg)\n",
    "        \n",
    "        if self.normalize:\n",
    "            out = F.normalize(out, p=2)\n",
    "        return out'''\n",
    "    \n",
    "    def forward(self,x,pos_edge_index,neg_edge_index,size = None):\n",
    "        out = None\n",
    "        x_j_pos = self.propagate(pos_edge_index,x=x)\n",
    "        x_j_neg = self.propagate(neg_edge_index,x=x)\n",
    "        out = self.lin_l(x) + self.lin_r(x_j_pos) - self.lin_r(x_j_neg)\n",
    "        if self.normalize:\n",
    "            out = F.normalize(out, p=2)\n",
    "        return out\n",
    "\n",
    "    def message(self, x_j):\n",
    "\n",
    "        out = None\n",
    "        out = x_j\n",
    "        return out\n",
    "\n",
    "    def aggregate(self, inputs, index, dim_size=None):\n",
    "        out = None\n",
    "        node_dim = self.node_dim\n",
    "        out = torch_scatter.scatter(inputs, index, dim=node_dim, reduce='mean')\n",
    "        return out\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "afe94c64-52de-4e2b-ab08-8644f28f122c",
   "metadata": {},
   "outputs": [],
   "source": [
    "class GAT(MessagePassing):\n",
    "\n",
    "    def __init__(self, in_channels, out_channels, heads = 2,\n",
    "                 negative_slope = 0.2, dropout = 0., **kwargs):\n",
    "        super(GAT, self).__init__(node_dim=0, **kwargs)\n",
    "\n",
    "        self.in_channels = in_channels\n",
    "        self.out_channels = out_channels\n",
    "        self.heads = heads\n",
    "        self.negative_slope = negative_slope\n",
    "        self.dropout = dropout\n",
    "\n",
    "        self.lin_l = None\n",
    "        self.lin_r = None\n",
    "        self.att_l = None\n",
    "        self.att_r = None\n",
    "\n",
    "        self.lin_l = nn.Linear(self.in_channels, self.heads*self.out_channels)\n",
    "        self.lin_r = self.lin_l\n",
    "        self.att_l = nn.Parameter(torch.zeros(self.heads, self.out_channels))\n",
    "        self.att_r=self.att_l\n",
    "        self.reset_parameters()\n",
    "\n",
    "    def reset_parameters(self):\n",
    "        nn.init.xavier_uniform_(self.lin_l.weight)\n",
    "        nn.init.xavier_uniform_(self.lin_r.weight)\n",
    "        nn.init.xavier_uniform_(self.att_l)\n",
    "        nn.init.xavier_uniform_(self.att_r)\n",
    "\n",
    "    def forward(self, x, edge_index, size = None):\n",
    "\n",
    "        H, C = self.heads, self.out_channels\n",
    "        x_l=self.lin_l(x).view(-1, H, C)\n",
    "        x_r=self.lin_r(x).view(-1, H, C)\n",
    "        alpha_l=self.att_l*x_l\n",
    "        alpha_r=self.att_r*x_r\n",
    "        out=self.propagate(edge_index, x=(x_l,x_r), alpha=(alpha_l, alpha_r), size=size).view(-1,H*C)\n",
    "        return out\n",
    "\n",
    "\n",
    "    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):\n",
    "        alpha=(alpha_j + alpha_i).squeeze(-1)\n",
    "        alpha=F.leaky_relu(alpha, negative_slope=0.2)\n",
    "        if ptr:\n",
    "            alpha = F.softmax(alpha,ptr)\n",
    "        else:\n",
    "            alpha=torch_geometric.utils.softmax(alpha, index)\n",
    "        alpha=F.dropout(alpha, p=self.dropout, training=self.training)\n",
    "        out=x_j * alpha\n",
    "        return out\n",
    "\n",
    "\n",
    "    def aggregate(self, inputs, index, dim_size = None):\n",
    "        node_dim = self.node_dim\n",
    "        out = torch_scatter.scatter(inputs, index, node_dim, dim_size=dim_size, reduce='sum')\n",
    "        return out"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "99809f80-3321-43d9-8330-b58f9dea92d0",
   "metadata": {},
   "outputs": [],
   "source": [
    "positive_edges=edge_index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "25efbdfd-962f-495e-b06e-2dac493ef197",
   "metadata": {},
   "outputs": [],
   "source": [
    "test=pd.read_csv('/Users/akshaj.g/Desktop/ml/social network /FacebookRecruiting/test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "7d0da67b-7fbc-4397-9156-e21ac24c44b9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([2, 9437519])"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_index = torch.tensor(test.values).T\n",
    "test_index.shape\n",
    "edge_index.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "321e435e-1a59-42be-afd0-d759dfc8e33b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from torch_geometric.utils import negative_sampling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "c84b4456-cae1-42af-96aa-aa9a55553711",
   "metadata": {},
   "outputs": [],
   "source": [
    "neg_edges=negative_sampling(edge_index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "19cb33a2-4cf4-4f1c-a051-b4e316a17e96",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([2, 9437519])"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "neg_edges.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "15c5832d-0d51-4b9c-ad78-43bbdc9da1ac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([200])"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "edge_label = torch.tensor(([1.] * 100 + [0.] * 100),requires_grad=True)\n",
    "edge_label.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "31ef4a26-14c3-493d-9a5a-5fcd73e31c8b",
   "metadata": {},
   "outputs": [],
   "source": [
    "max_node_index = max(positive_edges.max(), neg_edges.max())\n",
    "num_nodes = max_node_index + 1\n",
    "x = torch.rand(num_nodes, 16)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "341bb169-4835-4bfb-8211-dbf9bf77ce66",
   "metadata": {},
   "outputs": [],
   "source": [
    "loss_fn = nn.BCELoss()\n",
    "sigmoid = nn.Sigmoid()\n",
    "\n",
    "def accuracy(pred, label):\n",
    "  accu = 0.0\n",
    "  y_hat = (pred>0.5).type(torch.LongTensor)\n",
    "  accu = torch.mean((label==y_hat).type(torch.FloatTensor))\n",
    "  return accu"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "13fab26d-d37a-4d39-98fd-f5fffa241d98",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch.optim as optim\n",
    "\n",
    "def build_optimizer(args, params):\n",
    "    weight_decay = args.weight_decay\n",
    "    filter_fn = filter(lambda p : p.requires_grad, params)\n",
    "    if args.opt == 'adam':\n",
    "        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)\n",
    "    elif args.opt == 'sgd':\n",
    "        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)\n",
    "    elif args.opt == 'rmsprop':\n",
    "        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)\n",
    "    elif args.opt == 'adagrad':\n",
    "        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)\n",
    "    if args.opt_scheduler == 'none':\n",
    "        return None, optimizer\n",
    "    elif args.opt_scheduler == 'step':\n",
    "        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)\n",
    "    elif args.opt_scheduler == 'cos':\n",
    "        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)\n",
    "    return scheduler, optimizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "00d2e9b0-160e-4009-ae90-cff131b641fa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'import time\\nimport networkx as nx\\nimport numpy as np\\nimport torch\\nimport torch.optim as optim\\nfrom tqdm import trange\\nimport pandas as pd\\nimport copy\\n\\nfrom torch_geometric.datasets import TUDataset\\nfrom torch_geometric.datasets import Planetoid\\nfrom torch_geometric.data import DataLoader\\n\\nimport torch_geometric.nn as pyg_nn\\n\\nimport matplotlib.pyplot as plt\\n\\n\\ndef train(positive_edges,neg_edges,args,batch_size):\\n#building the model \\n\\n    model = GNNStack(16, args.hidden_dim, 8,args)\\n    \\n    scheduler, opt = build_optimizer(args, model.parameters())\\n\\n#starting the training \\n\\n    for epoch in trange(args.epochs, desc=\"Training\", unit=\"Epochs\"):\\n        total_loss = 0\\n        model.train()\\n        \\n#shuffle the edges\\n        p_idx = torch.randperm(positive_edges.size(1))\\n        p=positive_edges[:,p_idx]\\n        p=torch.tensor(p)\\n        n_idx = torch.randperm(neg_edges.size(1))\\n        n=neg_edges[:,n_idx]\\n        n=torch.tensor(n)\\n#batch the edges\\n        p_b = p[:,:batch_size]\\n        n_b = n[:,:batch_size]\\n        batch_edge = torch.cat([p_b,n_b], dim=1)\\n        print(batch_edge.shape)\\n#making pred and labels \\n        opt.zero_grad()\\n        pred = model(batch_edge)\\n        edge_label = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)], dim=0)\\n        train_edge_mask = batch_edge.train_mask\\n        train_pred = pred[train_edge_mask]\\n        train_labels = edge_label[train_edge_mask]\\n#loss is binary cross loss \\n            \\n        loss = F.binary_cross_entropy_with_logits(train_pred, train_labels.view(-1, 1))\\n        loss.backward()\\n        opt.step()\\n\\n        acc = accuracy(edge_feature , edge_label)\\n        print(emb)\\n        print(f\"Epoch: {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {acc}\")\\n    node_embeddings = emb.detach()\\n    return node_embeddings'"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''import time\n",
    "import networkx as nx\n",
    "import numpy as np\n",
    "import torch\n",
    "import torch.optim as optim\n",
    "from tqdm import trange\n",
    "import pandas as pd\n",
    "import copy\n",
    "\n",
    "from torch_geometric.datasets import TUDataset\n",
    "from torch_geometric.datasets import Planetoid\n",
    "from torch_geometric.data import DataLoader\n",
    "\n",
    "import torch_geometric.nn as pyg_nn\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "\n",
    "def train(positive_edges,neg_edges,args,batch_size):\n",
    "#building the model \n",
    "\n",
    "    model = GNNStack(16, args.hidden_dim, 8,args)\n",
    "    \n",
    "    scheduler, opt = build_optimizer(args, model.parameters())\n",
    "\n",
    "#starting the training \n",
    "\n",
    "    for epoch in trange(args.epochs, desc=\"Training\", unit=\"Epochs\"):\n",
    "        total_loss = 0\n",
    "        model.train()\n",
    "        \n",
    "#shuffle the edges\n",
    "        p_idx = torch.randperm(positive_edges.size(1))\n",
    "        p=positive_edges[:,p_idx]\n",
    "        p=torch.tensor(p)\n",
    "        n_idx = torch.randperm(neg_edges.size(1))\n",
    "        n=neg_edges[:,n_idx]\n",
    "        n=torch.tensor(n)\n",
    "#batch the edges\n",
    "        p_b = p[:,:batch_size]\n",
    "        n_b = n[:,:batch_size]\n",
    "        batch_edge = torch.cat([p_b,n_b], dim=1)\n",
    "        print(batch_edge.shape)\n",
    "#making pred and labels \n",
    "        opt.zero_grad()\n",
    "        pred = model(batch_edge)\n",
    "        edge_label = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)], dim=0)\n",
    "        train_edge_mask = batch_edge.train_mask\n",
    "        train_pred = pred[train_edge_mask]\n",
    "        train_labels = edge_label[train_edge_mask]\n",
    "#loss is binary cross loss \n",
    "            \n",
    "        loss = F.binary_cross_entropy_with_logits(train_pred, train_labels.view(-1, 1))\n",
    "        loss.backward()\n",
    "        opt.step()\n",
    "\n",
    "        acc = accuracy(edge_feature , edge_label)\n",
    "        print(emb)\n",
    "        print(f\"Epoch: {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {acc}\")\n",
    "    node_embeddings = emb.detach()\n",
    "    return node_embeddings'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "79aaf4f1-4adf-4a58-ad3c-310a878811db",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import time\n",
    "import networkx as nx\n",
    "import numpy as np\n",
    "import torch\n",
    "import torch.optim as optim\n",
    "from tqdm import trange\n",
    "import pandas as pd\n",
    "import copy\n",
    "\n",
    "from torch_geometric.datasets import TUDataset\n",
    "from torch_geometric.datasets import Planetoid\n",
    "from torch_geometric.data import DataLoader\n",
    "\n",
    "import torch_geometric.nn as pyg_nn\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def train(x, positive_edge, neg_edges, args, batch_size):\n",
    "    # Build the model\n",
    "    model = GNNStack(16, args.hidden_dim, 8, args)\n",
    "    \n",
    "    scheduler, opt = build_optimizer(args, model.parameters())\n",
    "\n",
    "    # Starting the training\n",
    "    for epoch in trange(args.epochs, desc=\"Training\", unit=\"Epochs\"):\n",
    "        total_loss = 0\n",
    "        model.train()\n",
    "\n",
    "        # Shuffle the edges\n",
    "        p_idx = torch.randperm(positive_edge.size(1))\n",
    "        p = positive_edge[:, p_idx]\n",
    "        n_idx = torch.randperm(neg_edges.size(1))\n",
    "        n = neg_edges[:, n_idx]\n",
    "        \n",
    "        num_batches = (positive_edge.size(1) + batch_size - 1) // batch_size\n",
    "        \n",
    "        for batch_idx in range(num_batches):\n",
    "            # Batch the edges\n",
    "            start = batch_idx * batch_size\n",
    "            end = min((batch_idx + 1) * batch_size, positive_edge.size(1))\n",
    "            pos_edge_index = p[:, start:end]\n",
    "            neg_edge_index = n[:, start:end]\n",
    "            # Calculate the maximum node index in the entire graph\n",
    "            max_node_index = max(pos_edge_index.max(), neg_edge_index.max())\n",
    "            num_nodes = max_node_index + 1\n",
    "            \n",
    "            # Create train masks with the shape of the entire graph\n",
    "            pos_edge_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.uint8)\n",
    "            neg_edge_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.uint8)\n",
    "\n",
    "            # Set the positive and negative edge masks to 1 where edges exist\n",
    "            pos_edge_mask[pos_edge_batch[0], pos_edge_index[1]] = 1\n",
    "            neg_edge_mask[neg_edge_batch[0], neg_edge_index[1]] = 1\n",
    "\n",
    "            # Make predictions and labels\n",
    "            opt.zero_grad()\n",
    "            pred = model(x, pos_edge_mask, neg_edge_mask)\n",
    "\n",
    "            # Filter predictions and labels using the train mask\n",
    "            train_pred = pred[train_edge_mask]\n",
    "            train_labels = edge_label[train_edge_mask]\n",
    "\n",
    "            # Calculate binary cross-entropy loss\n",
    "            loss = F.binary_cross_entropy_with_logits(train_pred, train_labels.view(-1, 1))\n",
    "            loss.backward()\n",
    "            opt.step()\n",
    "\n",
    "        acc = accuracy(edge_feature, edge_label)  \n",
    "        print(emb)\n",
    "        print(f\"Epoch: {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {acc}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "a0546496-d5f6-4162-987d-5f2fef73f7f2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'import time\\nimport networkx as nx\\nimport numpy as np\\nimport torch\\nimport torch.optim as optim\\nfrom tqdm import trange\\nimport pandas as pd\\nimport copy\\n\\nfrom torch_geometric.datasets import TUDataset\\nfrom torch_geometric.datasets import Planetoid\\nfrom torch_geometric.data import DataLoader\\n\\nimport torch_geometric.nn as pyg_nn\\n\\nimport matplotlib.pyplot as plt\\n\\ndef train(x, positive_edges, neg_edges, args, batch_size):\\n    # Build the model\\n    model = GNNStack(16, args.hidden_dim, 8, args)\\n    \\n    scheduler, opt = build_optimizer(args, model.parameters())\\n\\n    # Starting the training\\n    for epoch in trange(args.epochs, desc=\"Training\", unit=\"Epochs\"):\\n        total_loss = 0\\n        model.train()\\n        \\n        # Shuffle the edges\\n        p_idx = torch.randperm(positive_edges.size(1))\\n        p = positive_edges[:, p_idx]\\n        n_idx = torch.randperm(neg_edges.size(1))\\n        n = neg_edges[:, n_idx]\\n        \\n        # Batch the edges\\n        pos_edge_index = p[:, :batch_size]\\n        neg_edge_index = n[:, :batch_size]\\n        train_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)\\n        \\n        # Make predictions and labels\\n        opt.zero_grad()\\n        pred = model(x, train_edge_index, pos_edge_index, neg_edge_index)\\n        \\n        # Create the train mask\\n        train_edge_mask = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)], dim=0)\\n        train_edge_mask = train_edge_mask[train_edge_index]\\n        \\n        # Filter predictions and labels using the train mask\\n        train_pred = pred[train_edge_mask]\\n        train_labels = train_edge_mask\\n\\n        # Calculate binary cross-entropy loss\\n        loss = F.binary_cross_entropy_with_logits(train_pred, train_labels.view(-1, 1))\\n        loss.backward()\\n        opt.step()\\n\\n        acc = accuracy(train_pred, train_labels)\\n        print(f\"Epoch: {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {acc}\")\\n\\ntrained_node_embeddings = train(x, positive_edges, neg_edges, args, 10000)'"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "'''import time\n",
    "import networkx as nx\n",
    "import numpy as np\n",
    "import torch\n",
    "import torch.optim as optim\n",
    "from tqdm import trange\n",
    "import pandas as pd\n",
    "import copy\n",
    "\n",
    "from torch_geometric.datasets import TUDataset\n",
    "from torch_geometric.datasets import Planetoid\n",
    "from torch_geometric.data import DataLoader\n",
    "\n",
    "import torch_geometric.nn as pyg_nn\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def train(x, positive_edges, neg_edges, args, batch_size):\n",
    "    # Build the model\n",
    "    model = GNNStack(16, args.hidden_dim, 8, args)\n",
    "    \n",
    "    scheduler, opt = build_optimizer(args, model.parameters())\n",
    "\n",
    "    # Starting the training\n",
    "    for epoch in trange(args.epochs, desc=\"Training\", unit=\"Epochs\"):\n",
    "        total_loss = 0\n",
    "        model.train()\n",
    "        \n",
    "        # Shuffle the edges\n",
    "        p_idx = torch.randperm(positive_edges.size(1))\n",
    "        p = positive_edges[:, p_idx]\n",
    "        n_idx = torch.randperm(neg_edges.size(1))\n",
    "        n = neg_edges[:, n_idx]\n",
    "        \n",
    "        # Batch the edges\n",
    "        pos_edge_index = p[:, :batch_size]\n",
    "        neg_edge_index = n[:, :batch_size]\n",
    "        train_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)\n",
    "        \n",
    "        # Make predictions and labels\n",
    "        opt.zero_grad()\n",
    "        pred = model(x, train_edge_index, pos_edge_index, neg_edge_index)\n",
    "        \n",
    "        # Create the train mask\n",
    "        train_edge_mask = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)], dim=0)\n",
    "        train_edge_mask = train_edge_mask[train_edge_index]\n",
    "        \n",
    "        # Filter predictions and labels using the train mask\n",
    "        train_pred = pred[train_edge_mask]\n",
    "        train_labels = train_edge_mask\n",
    "\n",
    "        # Calculate binary cross-entropy loss\n",
    "        loss = F.binary_cross_entropy_with_logits(train_pred, train_labels.view(-1, 1))\n",
    "        loss.backward()\n",
    "        opt.step()\n",
    "\n",
    "        acc = accuracy(train_pred, train_labels)\n",
    "        print(f\"Epoch: {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {acc}\")\n",
    "\n",
    "trained_node_embeddings = train(x, positive_edges, neg_edges, args, 10000)'''\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "f2f9b444-7e65-47e2-90c1-95cab3ea4001",
   "metadata": {},
   "outputs": [],
   "source": [
    "class objectview:\n",
    "    def __init__(self, d):\n",
    "        self.__dict__ = d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2c2e6276-63ce-4490-8055-c38528ded1af",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Training:   0%|                                     | 0/500 [00:00<?, ?Epochs/s]"
     ]
    }
   ],
   "source": [
    "if 'IS_GRADESCOPE_ENV' not in os.environ:\n",
    "    for args in [\n",
    "        {'model_type': 'GraphSage', 'num_layers': 2, 'heads': 1, 'batch_size': 32, 'hidden_dim': 32, 'dropout': 0.5, 'epochs': 500, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.001},\n",
    "    ]:\n",
    "        args = objectview(args)\n",
    "        for model in ['GraphSage']:\n",
    "            args.model_type = model\n",
    "\n",
    "            # Match the dimension.\n",
    "            if model == 'GAT':\n",
    "                args.heads = 2\n",
    "            else:\n",
    "                args.heads = 1\n",
    "                \n",
    "trained_node_embeddings = train(x,positive_edges,neg_edges,args,1000)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "067a4c2b-4b7f-444b-bbc0-dc843980304d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
