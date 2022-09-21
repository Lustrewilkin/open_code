import dgl
import torch
import io
import numpy as np

def load_graph_mat(self, fre_chs,
                    feat_filename,edge_filename):
    feat_data = io.loadmat(feat_filename)  # 人X实验类型X实验次数X通道数量 X 时间序列
    feat_data = feat_data['dzq']
    lable = []

    edge_data = io.loadmat(edge_filename) # 人X实验类型 X实验次数X频率X连接权重矩阵    
    edge_data = edge_data['adj']
    edge = np.ones((18,18))
    edge_list = edge.nonzero()
    g = dgl.convert.graph(edge_list)

    graphs = []
    """
        由于数据是单人的，需要读取
        第一维度：[0,0],因为该维度封装了个人，进行人分类时需要扩展为i,在MATLAB里改就成，不过标签不一样了
        第二维度：[j,0],j是该类型实验具有的分类
        第三维度：[k,0],k是该实验的第几个trail
    """ 
    
    for i in range(feat_data.shape[0]):
        for j in range(feat_data[i,0].shape[0]):#特征
            for k in range(feat_data[i,0][j,0].shape[0]):
                f_d = feat_data[i,0][j,0][k,0]
                temp1 = np.c_[f_d,f_d,f_d]
                temp2 = temp1[:,0:512]
                lable.append([j])
                g.ndata['h'] =  torch.tensor(temp2).float()
                # 频率就先输入一个1个 , fre_chs 33x1 0-32选择
                temp3 = edge_data[i,0][j,0][k,0][fre_chs,0]
                g.edata['f'] = torch.tensor(temp3.reshape(-1,1)).float()
                graphs.append(g)

    return graphs, torch.tensor(lable)



