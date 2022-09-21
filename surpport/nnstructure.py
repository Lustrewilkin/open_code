from turtle import forward
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch.conv import EdgeWeightNorm, GraphConv
import numpy as np
import dgl
import torch

# classifier 实例化以后是一个字典，可以用 key 访问
# 第一个模型

# model 1


class Classifier1(nn.Module):

    def __init__(self, arg):
        super(Classifier1, self).__init__()
        # 初始化后定义网络结构
        # 有研究表明，即使不更新bias也不碍事，可以加速，少一半参数
        self.in_dim = arg.input_ft
        self.hidden_dim1 = arg.hidden_dim_1
        self.hidden_dim2 = arg.hidden_dim_2
        self.hidden_dim3 = arg.hidden_dim_3
        self.n_classes = arg.num_labels

        self.conv1 = nn.Conv1d(self.in_dim, 3, 3, 1, padding=2)
        self.gconv1 = GraphConv(self.in_dim, self.hidden_dim1,
                                norm='none', weight=True, bias=True)
        self.gconv2 = GraphConv(self.hidden_dim1, self.hidden_dim2,
                                norm='none', weight=True, bias=True)
        self.act_fc = nn.ReLU()
        self.classify1 = nn.Linear(self.hidden_dim2, self.hidden_dim3)
        self.classify2 = nn.Linear(self.hidden_dim3, self.n_classes)

    def forward(self, graph):

        #         norm = EdgeWeightNorm(norm='both')
        #         norm_edge_weight = norm(g, e_feat)

        # 应用图卷积和激活函数
        # h0 = self.conv1(graph.ndata['h'].float())
        h1 = self.gconv1(graph, graph.ndata['h'].float(
        ), edge_weight=graph.edata['f'].float())
        ac_h1 = self.act_fc(h1)
        h2 = self.gconv2(graph, ac_h1, edge_weight=graph.edata['f'].float())
        h = self.act_fc(h2)

        # 图读出阶段
        with graph.local_scope():
            graph.ndata['h'] = h
            # 使用平均读出计算图表示
            hg = dgl.readout_nodes(
                graph, 'h', weight=None, op='sum', ntype=None)
            # hg = self.classify1(hg)
            return self.classify1(hg)

    @property
    # 原文链接：https://blog.csdn.net/beilizhang/article/details/111570443
    # 自己定义num_labels
    def num_labels(self):
        return 5

    def __getitem__(self, idx):
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        return len(self.graphs)

# model 2
# oral type
# have some bug and unnecessary part


class Classifier2(nn.Module):
    def __init__(self, arg):
        super(Classifier2, self).__init__()
        self.in_dim = arg.input_ft
        self.hidden_dim1 = arg.hidden_dim_1
        self.n_classes = arg.num_labels

        self.conv1 = nn.Conv1d(
            128, 96, kernel_size=3, stride=1, padding=int((3-1)/2))
        self.conv2 = nn.Conv1d(
            128, 96, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(
            128, 96, kernel_size=3, stride=1, padding=3, dilation=3)

        self.pool = nn.MaxPool1d(kernel_size=39, stride=1, padding=3)

        # self.gconv1 = GraphConv(64, 32,
        #                         norm='none', weight=True, bias=True)
        # self.gconv2 = GraphConv(32, 16,
        #                         norm='none', weight=True, bias=True)

        self.gconv11 = GraphConv(96, 54,
                                 norm='none', weight=True, bias=True)
        self.gconv12 = GraphConv(96, 54,
                                 norm='none', weight=True, bias=True)
        self.gconv13 = GraphConv(96, 54,
                                 norm='none', weight=True, bias=True)

        self.gconv2 = GraphConv(54, 25,
                                norm='none', weight=True, bias=True)
        self.gconv3 = GraphConv(25, 13,
                                norm='none', weight=True, bias=True)
        self.gconv4 = GraphConv(25, 11,
                                norm='none', weight=True, bias=True)

        self.classify1 = nn.Linear(49, 27)
        self.classify2 = nn.Linear(27, self.n_classes)
        self.actf = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, graph):
        # conv1D 需要变形，所以要单独写个函数
        def conv1D(self, graph, convX):
            h0 = graph.ndata['h'].float().unsqueeze(0).permute(0, 2, 1)
            if convX == 1:
                h01 = self.conv1(h0)
                h02 = self.conv2(h0)
            elif convX == 2:
                h01 = self.conv2(h0)
                h02 = self.conv3(h0)
            elif convX == 3:
                h01 = self.conv3(h0)
                h02 = self.conv1(h0)
            h01 = h01.permute(0, 2, 1).squeeze(0)
            # h03 = h0.permute(0, 2, 1).squeeze(0)
            # h02 = self.dropout(h02)
            return self.actf(h01)+h02.permute(0, 2, 1).squeeze(0)

        def Norm2D(x):
            y = (x-torch.mean(x))/torch.sqrt(torch.var(x)+1e-5)
            return y
        # 普通1D卷积,TCN 不行
        h01 = conv1D(self, graph=graph, convX=1)
        h02 = conv1D(self, graph=graph, convX=2)
        h03 = conv1D(self, graph=graph, convX=3)

        # TCN
        # h0 = Tconv1D(self, graph, 1)

        # g_conv1
        h1_1 = self.gconv11(graph, h01  # graph.ndata['h'].float()
                            , edge_weight=graph.edata['f'].float())
        h1_2 = self.gconv12(graph, h02  # graph.ndata['h'].float()
                            , edge_weight=graph.edata['f'].float())
        h1_3 = self.gconv13(graph, h03  # graph.ndata['h'].float()
                            , edge_weight=graph.edata['f'].float())
        ac_h11 = self.actf(h1_1)
        ac_h12 = self.actf(h1_2)
        ac_h13 = self.actf(h1_3)

        ac_h1 = torch.add(torch.add(ac_h11, ac_h12), ac_h13)
        ac_h1 = self.dropout(ac_h1)

        # g_conv2
        h2 = self.gconv2(graph, ac_h1, edge_weight=graph.edata['f'].float())
        # h2 = self.dropout2(h2)
        ac_h2 = self.actf(h2)

        # g_conv3
        h3 = self.gconv3(graph, ac_h2, edge_weight=graph.edata['f'].float())
        # h3 = self.dropout2(h3)
        ac_h3 = self.actf(h3)

        # g_conv4
        h4 = self.gconv4(graph, ac_h2, edge_weight=graph.edata['f'].float())
        # h4 = self.dropout2(h4)
        ac_h4 = self.actf(h4)

        with graph.local_scope():
            # conv2 read
            graph.ndata['h'] = ac_h2  # + ac_h3
            hg2 = dgl.readout_nodes(
                graph, 'h', weight=None, op='sum', ntype=None)
            # conv3 read
            graph.ndata['h'] = ac_h3
            hg3 = dgl.readout_nodes(
                graph, 'h', weight=None, op='sum', ntype=None)

            # conv4 read
            graph.ndata['h'] = ac_h4
            hg4 = dgl.readout_nodes(
                graph, 'h', weight=None, op='sum', ntype=None)
            # return self.classify1(torch.cat((hg2, hg3, hg4), 1))
            return self.classify2(self.classify1(torch.cat((hg2, hg3, hg4), 1)))

    @ property
    def num_labels(self):
        return 5

    def __getitem__(self, idx):
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        return len(self.graphs)


# model 3
# transfomer
class Classifier3(nn.Module):
    def __init__(self, arg):
        super(Classifier3, self).__init__()
        self.in_dim = arg.input_ft
        self.hidden_dim1 = arg.hidden_dim_1
        self.n_classes = arg.num_labels

        encoder_ly = torch.nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=256)
        self.tf_ed = torch.nn.TransformerEncoder(encoder_ly, num_layers=6)

        self.conv = nn.Conv1d(
            128, 96, kernel_size=3, stride=1, padding=3, dilation=3)
        # self.gconv11 = GraphConv(128, 64,
        #                          norm='none', weight=True, bias=True)

        # self.gconv2 = GraphConv(64, 29,
        #                         norm='none', weight=True, bias=True)

        self.classify1 = nn.Linear(128, 64)
        self.classify2 = nn.Linear(64, self.n_classes)
        self.actf = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, graph):
        def conv1D(self, x):
            h0 = x.permute(0, 2, 1)
            h1 = self.conv(h0)
            h2 = h1.permute(0, 2, 1).squeeze(0)
            return torch.mul(self.actf(h2), self.tanh(h2))

        h0 = graph.ndata['h'].float().unsqueeze(0).permute(1, 0, 2)
        h01 = self.tf_ed(h0)
        h01 = h01.permute(1, 0, 2).squeeze(0)
        # h02 = conv1D(h01)
        # g_conv1
        # h1_1 = self.gconv11(graph, h01  # graph.ndata['h'].float()
        #                     , edge_weight=graph.edata['f'].float())
        # ac_h1 = self.actf(h1_1)
        # ac_h1 = self.dropout(ac_h1)

        # g_conv2
        # h2 = self.gconv2(graph, ac_h1, edge_weight=graph.edata['f'].float())
        # # h2 = self.dropout2(h2)
        # ac_h2 = self.actf(h2)

        with graph.local_scope():
            # conv2 read
            graph.ndata['h'] = h01  # + ac_h3
            hg2 = dgl.readout_nodes(
                graph, 'h', weight=None, op='sum', ntype=None)
        #     return self.classify2(self.classify1(hg2))
            return self.classify2(self.dropout2(self.classify1(hg2)))

    @ property
    def num_labels(self):
        return 5

    def __getitem__(self, idx):
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        return len(self.graphs)


# model 4
# ! it's already abandond
# cross + bn to reduce overfit
class Classifier4(nn.Module):
    def __init__(self, arg):
        super(Classifier4, self).__init__()
        self.n_classes = arg.num_labels

        self.conv1 = nn.Conv1d(
            128, 96, kernel_size=3, stride=1, padding=int((3-1)/2))
        self.conv2 = nn.Conv1d(
            128, 96, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(
            128, 96, kernel_size=3, stride=1, padding=3, dilation=3)

        self.gconv11 = GraphConv(96, 54,
                                 norm='none', weight=True, bias=True)
        self.gconv12 = GraphConv(96, 54,
                                 norm='none', weight=True, bias=True)
        self.gconv13 = GraphConv(96, 54,
                                 norm='none', weight=True, bias=True)

        self.pressConv = nn.Conv1d(3, 1, kernel_size=3, stride=1, padding=1)

        self.gconv2 = GraphConv(54, 25,
                                norm='none', weight=True, bias=True)
        self.gconv3 = GraphConv(25, 13,
                                norm='none', weight=True, bias=True)
        self.gconv4 = GraphConv(25, 11,
                                norm='none', weight=True, bias=True)

        self.classify1 = nn.Linear(49, 27)
        self.classify2 = nn.Linear(27, self.n_classes)
        self.actf = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, graph):
        # conv1D 需要变形，所以要单独写个函数
        def conv1D(self, graph, convX):
            h0 = graph.ndata['h'].float().unsqueeze(0).permute(0, 2, 1)
            if convX == 1:
                h01 = self.conv1(h0)
                h02 = self.conv2(h0)
            elif convX == 2:
                h01 = self.conv2(h0)
                h02 = self.conv3(h0)
            elif convX == 3:
                h01 = self.conv3(h0)
                h02 = self.conv1(h0)
            h01 = h01.permute(0, 2, 1).squeeze(0)
            # h03 = h0.permute(0, 2, 1).squeeze(0)
            # h02 = self.dropout(h02)
            return self.actf(h01)+h02.permute(0, 2, 1).squeeze(0)

        def Norm2D(x):
            y = (x-torch.mean(x))/torch.sqrt(torch.var(x)+1e-5)
            return y
        # 普通1D卷积,TCN 不行
        h01 = conv1D(self, graph=graph, convX=1)
        h02 = conv1D(self, graph=graph, convX=2)
        h03 = conv1D(self, graph=graph, convX=3)

        # g_conv1
        h1_1 = self.gconv11(graph, h01  # graph.ndata['h'].float()
                            , edge_weight=graph.edata['f'].float())
        h1_2 = self.gconv12(graph, h02  # graph.ndata['h'].float()
                            , edge_weight=graph.edata['f'].float())
        h1_3 = self.gconv13(graph, h03  # graph.ndata['h'].float()
                            , edge_weight=graph.edata['f'].float())
        ac_h11 = self.actf(h1_1).unsqueeze(0)
        ac_h12 = self.actf(h1_2).unsqueeze(0)
        ac_h13 = self.actf(h1_3).unsqueeze(0)  # node x emb

        ac_h1 = torch.cat([ac_h11, ac_h12, ac_h13], dim=0).permute(1, 0, 2)
        ac_h1 = self.pressConv(ac_h1).permute(1, 0, 2).squeeze(0)
        ac_h1 = self.dropout(ac_h1)

        # g_conv2
        h2 = self.gconv2(graph, ac_h1, edge_weight=graph.edata['f'].float())
        # h2 = self.dropout2(h2)
        ac_h2 = self.actf(h2)

        # g_conv3
        h3 = self.gconv3(graph, ac_h2, edge_weight=graph.edata['f'].float())
        # h3 = self.dropout2(h3)
        ac_h3 = self.actf(h3)

        # g_conv4
        h4 = self.gconv4(graph, ac_h2, edge_weight=graph.edata['f'].float())
        # h4 = self.dropout2(h4)
        ac_h4 = self.actf(h4)

        with graph.local_scope():
            # conv2 read
            graph.ndata['h'] = ac_h2  # + ac_h3
            hg2 = dgl.readout_nodes(
                graph, 'h', weight=None, op='sum', ntype=None)
            # conv3 read
            graph.ndata['h'] = ac_h3
            hg3 = dgl.readout_nodes(
                graph, 'h', weight=None, op='sum', ntype=None)

            # conv4 read
            graph.ndata['h'] = ac_h4
            hg4 = dgl.readout_nodes(
                graph, 'h', weight=None, op='sum', ntype=None)
            # return self.classify1(torch.cat((hg2, hg3, hg4), 1))
            return self.classify2(self.classify1(torch.cat((hg2, hg3, hg4), 1))), 0

    @ property
    def num_labels(self):
        return 5

    def __getitem__(self, idx):
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        return len(self.graphs)


# model 5
# no surper parameter to rec
class Eq_Classifer(nn.Module):
    def __init__(self, arg):
        super(Eq_Classifer, self).__init__()
        self.in_dim = arg.input_ft
        self.hidden_dim1 = arg.hidden_dim_1
        self.n_classes = arg.num_labels

        self.conv1 = nn.Conv1d(
            128, 128, kernel_size=3, stride=1, padding=int((3-1)/2))
        self.conv2 = nn.Conv1d(
            128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(
            128, 128, kernel_size=3, stride=1, padding=3, dilation=3)

        self.gconv11 = GraphConv(128, 128,
                                 norm='none', weight=True, bias=True)
        self.gconv12 = GraphConv(128, 128,
                                 norm='none', weight=True, bias=True)
        self.gconv13 = GraphConv(128, 128,
                                 norm='none', weight=True, bias=True)

        self.pressConv = nn.Conv1d(3, 1, kernel_size=3, stride=1, padding=1)

        self.gconv2 = GraphConv(128, 128,
                                norm='none', weight=True, bias=True)
        self.gconv3 = GraphConv(128, 128,
                                norm='none', weight=True, bias=True)
        self.gconv4 = GraphConv(128, 128,
                                norm='none', weight=True, bias=True)
        self.MaxPool1D = nn.MaxPool1d(10, 3, 5)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.classify1 = nn.Linear(128*3, 128)
        self.classify2 = nn.Linear(128, self.n_classes)
        self.actf = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, graph):
        # conv1D 需要变形，所以要单独写个函数
        def conv1D(self, graph, convX):
            h0 = graph.ndata['h'].float().unsqueeze(0).permute(0, 2, 1)
            if convX == 1:
                h01 = self.conv1(h0)
                h02 = self.conv2(h0)
            elif convX == 2:
                h01 = self.conv2(h0)
                h02 = self.conv3(h0)
            elif convX == 3:
                h01 = self.conv3(h0)
                h02 = self.conv1(h0)
            h01 = h01.permute(0, 2, 1).squeeze(0)
            # h03 = h0.permute(0, 2, 1).squeeze(0)
            # h02 = self.dropout(h02)
            return self.batchnorm1(self.actf(h01)+h02.permute(0, 2, 1).squeeze(0))

        def Norm2D(x):
            y = (x-torch.mean(x))/torch.sqrt(torch.var(x)+1e-5)
            return y
        # 普通1D卷积,TCN 不行
        h01 = conv1D(self, graph=graph, convX=1)
        h02 = conv1D(self, graph=graph, convX=2)
        h03 = conv1D(self, graph=graph, convX=3)

        # g_conv1
        h1_1 = self.gconv11(graph, h01  # graph.ndata['h'].float()
                            , edge_weight=graph.edata['f'].float())
        h1_2 = self.gconv12(graph, h02  # graph.ndata['h'].float()
                            , edge_weight=graph.edata['f'].float())
        h1_3 = self.gconv13(graph, h03  # graph.ndata['h'].float()
                            , edge_weight=graph.edata['f'].float())
        ac_h11 = self.actf(h1_1).unsqueeze(0)
        ac_h12 = self.actf(h1_2).unsqueeze(0)
        ac_h13 = self.actf(h1_3).unsqueeze(0)  # node x emb

        ac_h1 = torch.cat([ac_h11, ac_h12, ac_h13], dim=0).permute(1, 0, 2)
        ac_h1 = self.pressConv(ac_h1).permute(1, 0, 2).squeeze(0)
        ac_h1 = self.dropout(self.batchnorm1(ac_h1))

        # g_conv2
        h2 = self.gconv2(graph, ac_h1, edge_weight=graph.edata['f'].float())
        # h2 = self.dropout2(h2)
        ac_h2 = self.actf(h2)

        # g_conv3
        h3 = self.gconv3(graph, ac_h2, edge_weight=graph.edata['f'].float())
        # h3 = self.dropout2(h3)
        ac_h3 = self.actf(h3)

        # g_conv4
        h4 = self.gconv4(graph, ac_h3, edge_weight=graph.edata['f'].float())
        # h4 = self.dropout2(h4)
        ac_h4 = self.actf(h4)

        with graph.local_scope():
            # conv2 read
            graph.ndata['h'] = ac_h2  # + ac_h3
            hg2 = dgl.readout_nodes(
                graph, 'h', weight=None, op='sum', ntype=None)
            # conv3 read
            graph.ndata['h'] = ac_h3
            hg3 = dgl.readout_nodes(
                graph, 'h', weight=None, op='sum', ntype=None)

            # conv4 read
            graph.ndata['h'] = ac_h4
            hg4 = dgl.readout_nodes(
                graph, 'h', weight=None, op='sum', ntype=None)
            # return self.classify1(torch.cat((hg2, hg3, hg4), 1))
            # hg = self.MaxPool1D(torch.cat((hg2, hg3, hg4), 1))
            hg = torch.cat((hg2, hg3, hg4), 1)
            return self.classify2(self.classify1(hg))

    @ property
    def num_labels(self):
        return 5

    def __getitem__(self, idx):
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        return len(self.graphs)

# model 6
# singal pipline


class Classifier6(nn.Module):
    def __init__(self, arg) -> None:
        super(Classifier6, self).__init__()
        self.conv1 = nn.Conv1d(
            128, 128, kernel_size=3, stride=1, padding=int((3-1)/2))
        self.gconv1 = GraphConv(128, 96,
                                norm='none', weight=True, bias=True)

        self.gconv2 = GraphConv(96, 54,
                                norm='none', weight=True, bias=True)
        self.classify1 = nn.Linear(150, 5)
        self.actf = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, graph):
        h0 = graph.ndata['h'].float().unsqueeze(0).permute(0, 2, 1)
        h1 = self.actf(self.batchnorm1(
            self.conv1(h0).permute(0, 2, 1).squeeze(0)))

        h2 = self.gconv1(graph, h1, edge_weight=graph.edata['f'].float())
        h2 = self.actf(h2)

        h3 = self.gconv2(graph, h2, edge_weight=graph.edata['f'].float())
        h3 = self.dropout(self.actf(h3))

        with graph.local_scope():
            graph.ndata['h'] = torch.cat([h2, h3], dim=1)
            hg = dgl.readout_nodes(
                graph, 'h', weight=None, op='sum', ntype=None)
            return self.classify1(hg)

# model 7
# * cur use
# improve efficiency and simplify code


class Classifier7(nn.Module):
    def __init__(self, arg):
        super(Classifier7, self).__init__()

        self.n_classes = arg.num_labels
        self.pip_num = arg.pip_num
        self.dr = arg.dr  # drop_rate

        Conv = []
        Gcn = []
        for i in range(self.pip_num):
            Conv.append(nn.Conv1d(128, 96, kernel_size=3,
                        stride=1, padding=i+1, dilation=i+1))
            Gcn.append(GraphConv(96, 54, norm='none', weight=True, bias=True))
        self.ConvList = nn.ModuleList(Conv)
        self.GcnList = nn.ModuleList(Gcn)
        self.pressConv = nn.Conv1d(
            self.pip_num, 1, kernel_size=self.pip_num, stride=1, padding='same')
        self.Jk_GIN = nn.ModuleList([
            GraphConv(54, 34, norm='none', weight=True, bias=True),
            GraphConv(34, 25, norm='none', weight=True, bias=True),
            GraphConv(25, 16, norm='none', weight=True, bias=True),
        ])
        self.classify1 = nn.Linear(75, 120)
        self.classify2 = nn.Linear(120, self.n_classes)
        self.batchnorm1 = nn.BatchNorm1d(96)
        self.batchnorm2 = nn.BatchNorm1d(54)
        self.batchnorm3 = nn.BatchNorm1d(75)
        self.actf = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=arg.dr)

    def forward(self, graph: dgl.DGLGraph):
        emd_h = self.BN_embedding(graph)
        emd_h = self.dropout(emd_h)
        logits = self.Presentation(graph, emd_h)
        return logits, emd_h

    def cross_connect(self, h01, h02) -> torch.Tensor:
        h01 = h01.permute(0, 2, 1).squeeze(0)
        return self.batchnorm1(self.actf(h01)+h02.permute(0, 2, 1).squeeze(0))

    # Brain Network embedding
    def BN_embedding(self, graph: dgl.DGLGraph) -> torch.Tensor:
        # conv1D 需要变形，所以要单独写个函数
        # 普通1D卷积, TCN 不合适
        h0 = graph.ndata['h'].float().unsqueeze(
            0).permute(0, 2, 1)  # shape = [1, 128, 18*n]
        h0x = []
        for i in range(self.pip_num):
            h0x.append(self.ConvList[i](h0))

        edeg_w = graph.edata['f'].float()
        h2x = []
        for i in range(self.pip_num):
            h1x = self.cross_connect(
                h0x[i % self.pip_num], h0x[(i+1) % self.pip_num])
            temp = self.GcnList[i](graph, h1x, edge_weight=edeg_w)
            h2x.append(self.actf(temp).unsqueeze(0))  # reback to node x emb

        ac_h1 = self.pressConv(torch.cat(h2x, dim=0).permute(
            1, 0, 2)).permute(1, 0, 2).squeeze(0)
        # without dropout, it's indivual for module
        ac_h1 = self.batchnorm2(ac_h1)

        return ac_h1

    # presetation module
    def Presentation(self, graph: dgl.DGLGraph, emd_h: torch.Tensor) -> torch.Tensor:
        h0x = [emd_h]
        edeg_w = graph.edata['f'].float()
        for i in range(len(self.Jk_GIN)):
            h0x.append(self.actf(self.Jk_GIN[i](
                graph, h0x[i], edge_weight=edeg_w)))

        with graph.local_scope():
            ac_h = torch.cat(h0x[1:], 1)
            graph.ndata['h'] = ac_h
            hg = dgl.readout_nodes(graph, 'h', None, op='sum', ntype=None)
            return self.classify2(self.classify1(self.batchnorm3(hg)))
