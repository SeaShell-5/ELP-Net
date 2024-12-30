import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
from Baselines.Net.GNN import gtnet
from Baselines.Net.Transnet import EncoderLayer



class ELPNet(torch.nn.Module):
    def __init__(
            self,
            inplanes,
            midplanes,
            outplanes,
            in_vector,
            out_vector,
            in_rnn,
            out_rnn,
            outdense,
            gnn_dim,
            Trans_module,
            GNN_module,
            node_num=12,
            num_heads=8,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            Trans_layers=1,
            GNN_layers=1,
    ):
        super(ELPNet, self).__init__()
        self.midplanes = midplanes
        self.outplanes = outplanes
        self.out_vector = out_vector
        self.plane_length = outdense  # 地图的长 * 宽
        self.out_rnn = out_rnn
        self.inplanes = inplanes
        self.node_num = node_num
        self.h = None
        self.Trans_module = Trans_module
        self.GNN_module = GNN_module
        

        self.conv0 = nn.Conv2d(in_channels=inplanes, out_channels=midplanes, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(midplanes)

        self.conv1 = nn.Conv2d(in_channels=midplanes, out_channels=midplanes, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes)

        self.conv2 = nn.Conv2d(in_channels=midplanes, out_channels=node_num, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(node_num)
        # self.conv3 = nn.Conv2d(in_channels=midplanes, out_channels=outplanes, stride=1, kernel_size=3, padding=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(outplanes)

        self.linear_g = nn.Linear(in_vector, out_vector)  

        self.self_attention_norm = nn.LayerNorm(out_vector)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(out_vector)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        self.layernorm = nn.LayerNorm(out_vector)

        encoders = [EncoderLayer(out_vector, out_vector * 2, dropout_rate, attention_dropout_rate, num_heads) 
                         for _ in range(Trans_layers)]
        self.Trans_layers = nn.ModuleList(encoders)
        self.gtnet = gtnet(gcn_true=True, buildA_true=False, gcn_depth=2, num_nodes=12, 
                           in_dim=outdense, residual_channels=gnn_dim, layers=GNN_layers)

        for num in range(6):
            setattr(self, "pre_fc{}".format(num), nn.Linear(node_num * gnn_dim + out_vector, in_rnn))
            setattr(self, "rnn{}".format(num), nn.GRUCell(input_size=in_rnn, hidden_size=out_rnn))
            setattr(self, "end_fc{}".format(num), nn.Linear(out_rnn, outdense))
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, states_G, states_S, require_init):
        batch = states_S.size(1)
        seq_len = states_S.size(0)
        self.h = [None for _ in range(6)]
        for num in range(6):
            if self.h[num] is None:
                self.h[num] = Variable(states_S.data.new().resize_((batch, self.out_rnn)).zero_())
        if True in require_init:
            for idx, init in enumerate(require_init):
                if init:
                    for num in range(6):
                        self.h[num][idx, :].zero_()

        values = []

        x_G = self.linear_g(states_G.transpose(0, 1))
        # 是否添加Transformer模块
        if self.Trans_module:
            for enc_layer in self.Trans_layers:
                x_G = enc_layer(x_G)
        
        # 对兵棋的态势图进行变换
        x_S = states_S.view(seq_len, batch, -1, self.plane_length)
        x_S = F.relu(self.bn0(self.conv0(x_S.permute(1, 2, 3, 0))))
        x_S = F.relu(self.bn1(self.conv1(x_S)))
        x_S = F.relu(self.bn2(self.conv2(x_S)))
        
        if self.GNN_module:
            x_S = self.gtnet(x_S.transpose(1, 2))
            
        
        for idx, (x_s, x_g) in enumerate(zip(x_S.permute(3, 0, 2, 1), x_G.transpose(0, 1))):
            x_s = x_s.reshape(-1, x_s.shape[1] * x_s.shape[2])
            x = torch.cat((x_s, x_g), 1)
            
            x_list = [F.relu(getattr(self, "pre_fc{}".format(num))(x)) for num in range(6)]
            self.h = [getattr(self, "rnn{}".format(num))(x_list[num], self.h[num]) for num in range(6)]
            x_list = [F.relu(getattr(self, "end_fc{}".format(num))(self.h[num])) for num in range(6)]
            x = torch.stack(x_list, dim=1)
            probas = self.logsoftmax(x).exp()
            values.append(probas)  

        return values

    def detach(self):
        for num in range(6):
            if self.h[num] is not None:
                self.h[num].detach_()






