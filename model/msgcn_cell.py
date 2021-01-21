import numpy as np
import torch
import torch.nn.functional as F

from lib import utils

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")


class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class MSGGRUCell(torch.nn.Module):
    def __init__(self, num_units, adj_mx, mask, num_nodes, nonlinearity='tanh'):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._supports = []
        for i in range(8):
            self._supports.append(mask[i])
        
        self._gconv_params = LayerParams(self, 'gconv')


    def forward(self, inputs, hx, adj):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        output_size = 2 * self._num_units
        value = torch.sigmoid(self._gconv(inputs, hx, output_size, adj, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, r * hx, self._num_units, adj)

        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    @staticmethod
    def _concat_f(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=-1)

    def _gconv(self, inputs, state, output_size, adj, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)
        # X^t [B,N,F]  H^t-1 [B,N,F]
        # x [B,N,2*F]

        x = inputs_and_state

        x0 = x.permute(1, 2, 0)  # [N,F,B]
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size]) # [N,F*B]
        x = torch.unsqueeze(x0, 0) # x和x0现在都是[N,F*B]而且x=x0

        
        x_tmp = torch.matmul(adj[0], x0)
        x = self._concat(x, x_tmp)
        #GCN AXW
        for i, support in enumerate(self._supports):
            if i < 4:
                x1 = torch.matmul(support*adj[0], x0) #support*adj[0]取出子图 support [N,N] adj[0] [N,N] x0[N,F*B]    AX
                x = self._concat(x, x1) 
            else:
                x1 = torch.matmul(support*adj[1], x0)
                x = self._concat(x, x1)
        x_tmp = torch.matmul(adj[1], x0)
        x = self._concat(x, x_tmp)

            # x [N,9*F*B]

                # for k in range(2, self._max_diffusion_step + 1):
                #     x2 = 2 * torch.sparse.mm(support, x1) - x0
                #     x = self._concat(x, x2)
                #     x1, x0 = x2, x1
        
        num_matrices = len(self._supports) + 1 + 2  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size]) #[9,N,F,B]

        x_bp = x[0,...].permute(2,0,1).reshape(batch_size, self._num_nodes, -1) #取出第一个元素做残差连接
        x_i = x[1:6,...] #取出第2-5个元素 是原始空间的元素
        x_s = x[6:,...] #隐空间

        x_i = x_i.permute(3,1,0,2)
        x_s = x_s.permute(3,1,0,2)

        weights_s = self._gconv_params.get_weights((input_size, output_size))
        x_s = torch.matmul(x_s, weights_s)

        weights_i = self._gconv_params.get_weights((input_size, output_size))
        x_i = torch.matmul(x_i, weights_i)

        weights_bp = self._gconv_params.get_weights((input_size, output_size))
        x_bp = torch.matmul(x_bp, weights_bp)

        ###concat
        # x_s = torch.reshape(x_s, shape=[batch_size * self._num_nodes, 5*output_size]) #[B*N, 4*F]
        # x_i = torch.reshape(x_i, shape=[batch_size * self._num_nodes, 5*output_size]) #[B*N, 4*F]
        # weights_ss = self._gconv_params.get_weights((5*output_size, output_size))
        # x_s = torch.matmul(x_s, weights_ss)
        # weights_ii = self._gconv_params.get_weights((5*output_size, output_size))
        # x_i = torch.matmul(x_i, weights_ii)
        ###

        ### no share weights
        # tmps = []
        # tmpi = []
        # for i in range(4):
        #     weights_ts = self._gconv_params.get_weights((input_size, output_size))
        #     tmps.append(torch.matmul(x_s[:,:,i:i+1,:], weights_ts))
            

        #     weights_ti = self._gconv_params.get_weights((input_size, output_size))
        #     tmpi.append(torch.matmul(x_i[:,:,i:i+1,:], weights_ti))

        # x_s = torch.cat(tmps, 2)
        # x_i = torch.cat(tmpi, 2)

        # weights_bp = self._gconv_params.get_weights((input_size, output_size))
        # x_bp = torch.matmul(x_bp, weights_bp)
        ###

        ###max
        # x_s, _ = torch.max(x_s, 2)
        # x_i, _ = torch.max(x_i, 2)
        ###

        ###mean
        # x_s = torch.mean(x_s, 2)
        # x_i = torch.mean(x_i, 2)
        ###

        ###sum
        # x_s = torch.sum(x_s, 2)
        # x_i = torch.sum(x_i, 2)
        ###

        ###attention
        weights_atts = self._gconv_params.get_weights((self._num_nodes, output_size, output_size))
        weights_atts1 = self._gconv_params.get_weights((self._num_nodes, output_size, 1))

        atts = F.relu(torch.einsum('ijkl,jlb->ijkb',x_s, weights_atts))
        atts = torch.einsum('ijkl,jlb->ijkb',atts, weights_atts1).permute(0,1,3,2)
        # atts = F.relu(torch.matmul(x_s, weights_atts))
        # atts = torch.matmul(atts, weights_atts1).permute(0,1,3,2)
        atts = F.softmax(atts, -1)

        weights_atti = self._gconv_params.get_weights((self._num_nodes, output_size, output_size))
        weights_atti1 = self._gconv_params.get_weights((self._num_nodes, output_size, 1))
        atti = F.relu(torch.einsum('ijkl,jlb->ijkb',x_i, weights_atti))
        atti = torch.einsum('ijkl,jlb->ijkb',atti, weights_atti1).permute(0,1,3,2)
        # atti = F.relu(torch.matmul(x_i, weights_atti))
        # atti = torch.matmul(atti, weights_atti1).permute(0,1,3,2)
        atti = F.softmax(atti, -1)

        x_i = torch.matmul(atti, x_i).squeeze(2)
        x_s = torch.matmul(atts, x_s).squeeze(2)
        ###

        ##all att
        # x_bp = x[0,...].permute(2,0,1) #取出第一个元素做残差连接
        # x_i = x[1:,...] #取出第2-5个元素 是原始空间的元素
        # # x_s = x[5:,...] #隐空间

        # x_i = x_i.permute(3,1,0,2)
        # # x_s = x_s.permute(3,1,0,2)

        # # weights_s = self._gconv_params.get_weights((input_size, output_size))
        # # x_s = torch.matmul(x_s, weights_s)

        # weights_i = self._gconv_params.get_weights((input_size, output_size))
        # x_i = torch.matmul(x_i, weights_i)

        # weights_bp = self._gconv_params.get_weights((input_size, output_size))
        # x_bp = torch.matmul(x_bp, weights_bp)

        # # weights_atts = self._gconv_params.get_weights((self._num_nodes, output_size, output_size))
        # # weights_atts1 = self._gconv_params.get_weights((self._num_nodes, output_size, 1))

        # # atts = torch.einsum('ijkl,jlb->ijkb',x_s, weights_atts)
        # # atts = torch.einsum('ijkl,jlb->ijkb',atts, weights_atts1).permute(0,1,3,2)
        # # atts = F.softmax(atts, -1)

        # weights_atti = self._gconv_params.get_weights((self._num_nodes, output_size, output_size))
        # weights_atti1 = self._gconv_params.get_weights((self._num_nodes, output_size, 1))
        # atti = torch.einsum('ijkl,jlb->ijkb',x_i, weights_atti)
        # atti = torch.einsum('ijkl,jlb->ijkb',atti, weights_atti1).permute(0,1,3,2)
        # atti = F.softmax(atti, -1)

        # x_i = torch.matmul(atti, x_i).squeeze(2)
        # x = torch.cat([x_bp,x_i], -1) #[B*N,3*F'] 再做一个全连接wx+b
        # weights = self._gconv_params.get_weights((output_size * 2, output_size)) #全连接的W [3*F',F'']
        # x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size) [B*N,F'']

        # biases = self._gconv_params.get_biases(output_size, bias_start)
        # x += biases
        # x_s = torch.matmul(atts, x_s).squeeze(2)
        ####

        ###concat
        # num_matrices = 3  # Adds for x itself.
        # x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size]) #[9,N,F,B]

        # x_bp = x[0,...].permute(2,0,1) #取出第一个元素做残差连接
        # x_i = x[1,...] #取出第2-5个元素 是原始空间的元素
        # x_s = x[2,...] #隐空间
        # x_s = x_s.permute(2, 0, 1)  # (batch_size, num_nodes, input_size, order) [b,n,f,4]
        # x_s = torch.reshape(x_s, shape=[batch_size * self._num_nodes, input_size]) #[B*N, 4*F]

        # weights_s = self._gconv_params.get_weights((input_size, output_size)) #[F*4, F'] GCN 的 W
        # x_s = torch.matmul(x_s, weights_s)  # (batch_size * self._num_nodes, output_size) x_s = AX 现在 x_s:[B*N,F']

        # biases_s = self._gconv_params.get_biases(output_size, bias_start)
        # x_s += biases_s

        # # 隐空间的GCN x_s = \sigma_{i=1}^4 A_iXW    A_i = sub_i * (A_s * A_sadp)

        # # x_bp = x_bp.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        # x_bp = torch.reshape(x_bp, shape=[batch_size * self._num_nodes, input_size * 1])

        # weights_bp = self._gconv_params.get_weights((input_size * 1, output_size))
        # x_bp = torch.matmul(x_bp, weights_bp)  # (batch_size * self._num_nodes, output_size)

        # biases_bp = self._gconv_params.get_biases(output_size, bias_start)
        # x_bp += biases_bp

        # x_i = x_i.permute(2, 0, 1)  # (batch_size, num_nodes, input_size, order)
        # x_i = torch.reshape(x_i, shape=[batch_size * self._num_nodes, input_size])

        # weights_i = self._gconv_params.get_weights((input_size, output_size))
        # x_i = torch.matmul(x_i, weights_i)  # (batch_size * self._num_nodes, output_size)

        # biases_i = self._gconv_params.get_biases(output_size, bias_start)
        # x_i += biases_i
        ###

        # # 原始空间的GCN x_i = \sigma_{i=1}^4 A_iXW    A_i = sub_i * (A_init * A_initadp)

        # # x_i x_s shape [B*N,F'] 加上之前拿来做残差连接的x_bp

        
        x = torch.cat([x_i,x_s,x_bp], -1) #[B*N,3*F'] 再做一个全连接wx+b
        # x = torch.cat([x_bp,x_i], -1) #[B*N,3*F'] 再做一个全连接wx+b
        weights = self._gconv_params.get_weights((output_size * 3, output_size)) #全连接的W [3*F',F'']
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size) [B*N,F'']

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])

        ###attention cat
        # x = torch.stack([x_bp,x_i,x_s], 1) #[B*N,3*F'] 再做一个全连接wx+b
        # # x = torch.cat([x_bp,x_i], -1) #[B*N,3*F'] 再做一个全连接wx+b
        # weights = self._gconv_params.get_weights((output_size, output_size)) #全连接的W [3*F',F'']
        # weights1 = self._gconv_params.get_weights((output_size, 1)) #全连接的W [3*F',F'']
        # att = F.relu(torch.matmul(x, weights))  # (batch_size * self._num_nodes, output_size) [B*N,F'']
        # att = F.softmax(torch.matmul(x, weights1), -1).permute(0,2,1)

        # x = torch.matmul(att, x).squeeze(1)

        # weights = self._gconv_params.get_weights((output_size, output_size)) #全连接的W [3*F',F'']
        # x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size) [B*N,F'']
        # biases = self._gconv_params.get_biases(output_size, bias_start)
        # x += biases
        # # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        # return torch.reshape(x, [batch_size, self._num_nodes * output_size])
        ###

        ###weights fusion
        # x = torch.cat([x_bp,x_i,x_s], -1) #[B*N,3*F'] 再做一个全连接wx+b
        # # x = torch.cat([x_bp,x_i], -1) #[B*N,3*F'] 再做一个全连接wx+b
        # weights = self._gconv_params.get_weights((output_size, output_size)) #全连接的W [3*F',F'']
        # x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size) [B*N,F'']

        # biases = self._gconv_params.get_biases(output_size, bias_start)
        # x += biases
        # # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        # x = x_bp+x_i+x_s
        # return torch.reshape(x, [batch_size, self._num_nodes * output_size])
        ###


        # x_bp = x[0,...].permute(2,0,1)
        # x_i = x[1:5,...]
        # # x_s = x[5:,...]
        # # num_matrices_i = 11  # Adds for x itself.
        # # num_matrices_s = 8
        # # x_i = torch.reshape(x_i, shape=[num_matrices_i, self._num_nodes, input_size, batch_size])
        # x_i = x_i.permute(3, 1, 0, 2)  # (batch_size, num_nodes, num_matrices, input_size)
        # # x_s = torch.reshape(x_s, shape=[num_matrices_s, self._num_nodes, input_size, batch_size])
        # # x_s = x_s.permute(3, 1, 0, 2)  # (batch_size, num_nodes, input_size, num_matrices)
        # # x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])
        # weights_trans_i = self._gconv_params.get_weights((self._num_nodes, input_size, output_size))
        # # weights_trans_s = self._gconv_params.get_weights((self._num_nodes, input_size, output_size))
        # weights_att_i = self._gconv_params.get_weights((self._num_nodes, output_size, 1))
        # # weights_att_s = self._gconv_params.get_weights((self._num_nodes, output_size, 1))
        # att_i = torch.einsum('ijkl,jlb->ijkb',x_i, weights_trans_i)
        # att_i = torch.einsum('ijkl,jlb->ijkb',att_i, weights_att_i)
        # att_i = torch.squeeze(att_i, -1)
        # att_i = F.softmax(att_i, -1)
        # att_i = torch.unsqueeze(att_i, 2)
        # x_i = torch.einsum('ijkl,ijlb->ijkb',att_i, x_i)
        # x_i = torch.reshape(x_i, shape=[batch_size, self._num_nodes, input_size])

        # # att_s = torch.einsum('ijkl,jlb->ijkb',x_s, weights_trans_s)
        # # att_s = torch.einsum('ijkl,jlb->ijkb',att_s, weights_att_s)
        # # att_s = torch.squeeze(att_s, -1)
        # # att_s = F.softmax(att_s, -1)
        # # att_s = torch.unsqueeze(att_s, 2)
        # # x_s = torch.einsum('ijkl,ijlb->ijkb',att_s, x_s)
        # # x_s = torch.reshape(x_s, shape=[batch_size, self._num_nodes, input_size])

        # x = torch.cat([x_bp, x_i], dim=-1)

        # weights = self._gconv_params.get_weights((2*input_size, output_size))
        # x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        # biases = self._gconv_params.get_biases(output_size, bias_start)
        # x += biases
        # # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        # return torch.reshape(x, [batch_size, self._num_nodes * output_size])
