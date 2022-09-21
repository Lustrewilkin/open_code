from dgl.dataloading import GraphDataLoader
from scipy.io import loadmat
import numpy as np
import os
import torch
import dgl
from dgl.data.dgl_dataset import DGLDataset
from dgl.data.utils import download, save_graphs, load_graphs, \
    check_sha1, deprecate_property
from surpport.Args import Args
import surpport.myfunction as MF
#from ..convert import graph as dgl_graph

# 想想在写


def dataload(arg, model, opt):
    dataset = my_dataset(arg, name=arg.s_name, raw_dir=arg.raw_data_dir,
                         save_dir=arg.save_data_dir)
    if not arg.new_train:
        MF.continue_tr(arg.dir, model, opt, arg)
    else:
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])
        arg.tr_id = train_dataset.indices
        arg.ts_id = test_dataset.indices

    train_dataset = torch.utils.data.Subset(
        dataset=dataset, indices=arg.tr_id)
    test_dataset = torch.utils.data.Subset(
        dataset=dataset, indices=arg.ts_id)
    tr_dataloader = dataloader(
        train_dataset, arg.batch_size, collate=MF.collate, shuffle=True)
    ts_dataloader = dataloader(
        test_dataset, arg.batch_size, collate=MF.collate, shuffle=True)
    return arg, tr_dataloader, ts_dataloader


def dataloader(dataset, batch_size, collate, shuffle):
    gdataloader = GraphDataLoader(
        dataset,
        collate_fn=collate,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
        # sampler = sampler
    )
    return gdataloader


class my_dataset(DGLDataset):
    """ 用于在DGL中自定义图数据集的模板：

    Parameters
    ----------
    url : str
        下载原始数据集的url。
    raw_dir : str
        指定下载数据的存储目录或已下载数据的存储目录。默认: ~/.dgl/
    save_dir : str
        处理完成的数据集的保存目录。默认：raw_dir指定的值
    force_reload : bool
        是否重新导入数据集。默认：False
    verbose : bool
        是否打印进度信息。
    """

    def __init__(self,
                 arg,
                 name=None,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False
                 ):
        self.arg1 = arg
        super(my_dataset, self).__init__(name=name,
                                         url=url,
                                         raw_dir=raw_dir,
                                         save_dir=save_dir,
                                         force_reload=force_reload,
                                         verbose=verbose
                                         )

    """
    如果用户的数据集已经在本地磁盘中，请确保它被存放在目录 raw_dir 中。 
    ---
    如果用户想在任何地方运行代码而又不想自己下载数据并将其移动到正确的目录中，
    则可以通过实现函数 download() 来自动完成。  
    ---
    如果加载OGB，从下文摘取
    https://docs.dgl.ai/en/latest/guide_cn/data-loadogb.html
    """

    def process(self):
        adj_path = self.raw_dir + '\\adjoridata\\' + self.name + '.npy'
        sig_path = self.raw_dir + '\\signal\\signal.npy'
        self.graphs, self.label = self.load_graph(sig_path, adj_path)

    # def load_graph_mat(self, filename):
    #     data = io.loadmat(filename)
    #     labels = F.tensor(data['T'], dtype=F.data_type_dict['float32'])
    #     feats = data['X']
    #     num_graphs = labels.shape[0]
    #     graphs = []
    #     for i in range(num_graphs):
    #         edge_list = feats[i].nonzero()
    #         g = dgl.graph(edge_list)
    #         g.edata['h'] = F.tensor(feats[i][edge_list[0], edge_list[1]].reshape(-1, 1),
    #                                 dtype=F.data_type_dict['float32'])
    #         graphs.append(g)
    #     return graphs, labels
    def load_graph(self, sig_dir, adj_dir):

        # data = io.loadmat(raw_dir)  # 人X实验类型X实验次数X [通道数量x时间序列]
        # n_data = data['data'][0, 0]  # 5x1
        # # 人X实验类型 X实验次数X频率X连接权重矩阵
        # e_data = data['data'][1, 0]  # 5x1

        n_data = np.load(sig_dir, allow_pickle=True)
        for i in range(12):
            if self.arg1.patient[i] == self.name:
                n_data = n_data[i, 0]
        e_data = np.load(adj_dir, allow_pickle=True)
        fre_chs = self.arg1.fre_chs
        labels = []
        graphs = []
        """
            由于数据是单人的，需要读取
            第一维度：[0,0],因为该维度封装了个人，进行人分类时需要扩展为i,在MATLAB里改就成，不过标签不一样了
            第二维度：[j,0],j是该类型实验具有的分类
            第三维度：[k,0],k是该实验的第几个trail
        """
        for j in range(n_data.shape[0]):  # 分类数目
            n1 = n_data[j, 0]
            e1 = e_data[j]
            count = 125
            k, k1 = 0, 0
            while k < count:  # 摘取125个样例
                if k1 < n1.shape[0]:  # trail
                    f_d = n1[k1, 0]
                    temp1 = np.c_[f_d, f_d, f_d, f_d]  # concat
                    orp = 0
                    drp = 128
                    for l in range(3):
                        labels.append([j])
                        # 创建图
                        e_d = e1[k1, 0][0, 0][fre_chs, 0]
                        if e_d.shape != (18, 18):
                            e_d = e1[k1-1, 0][0, 0][fre_chs, 0]
                        e_d = torch.tensor(e_d)
                        e_d1 = e_d.reshape(-1)
                        edge_cout = torch.zeros_like(e_d1)
                        values, indices = e_d1.topk(int(
                            e_d1.shape[0]/4 + e_d.shape[0])-1, dim=0, largest=True, sorted=True)  # +18是补一个矩阵长度
                        edge_cout = edge_cout.scatter(
                            dim=0, index=indices, value=1)

                        edge_cout = edge_cout.reshape(
                            e_d.shape[0], e_d.shape[1])
                        edge_list = edge_cout.nonzero()
                        g = dgl.convert.graph(edge_list.tolist())

                        temp2 = temp1[:, orp:drp]
                        g.ndata['h'] = torch.tensor(temp2).float()
                        orp, drp = orp+128, drp+128

                        # 频率就先输入一个1个 , fre_chs 33x1 0-32选择
                        # e_d = torch.mul(e_d,edge_cout)
                        g.edata['f'] = values.reshape(-1, 1).float()
                        graphs.append(g)
                    k1 += 1
                    k += 1
                else:
                    k1 = k1-n1.shape[0]
            # temp2 = temp1[:,0:512]
            # temp3 = temp1[:,128:256]
            # temp4 = temp1[:,256:384]
            # temp5 = temp1[:,384:512]
        return graphs, torch.tensor(labels)

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(
            self.save_path, 'f{0}_dgl_graph.bin'.format(self.arg1.fre_chs))
        save_graphs(str(graph_path), self.graphs, {'labels': self.label})

    def has_cache(self):
        graph_path = os.path.join(
            self.save_path, 'f{0}_dgl_graph.bin'.format(self.arg1.fre_chs))
        #graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def load(self):
        graphs, label_dict = load_graphs(
            os.path.join(self.save_path, 'f{0}_dgl_graph.bin'.format(self.arg1.fre_chs)))
        self.graphs = graphs
        self.label = label_dict['labels']

    def download(self):
        pass
        # file_path = os.path.join(self.raw_dir, self.name + '.mat')
        # download(self.url, path=file_path)
        # if not check_sha1(file_path, self._sha1_str):
        #     raise UserWarning('File {} is downloaded but the content hash does not match.'
        #                       'The repo may be outdated or download may be incomplete. '
        #                       'Otherwise you can create an issue for it.'.format(self.name))

    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return 5

    def __getitem__(self, idx):
        r""" Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (:class:`dgl.DGLGraph`, Tensor)
        """
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        r"""Number of graphs in the dataset.

        Return
        -------
        int
        """
        return len(self.graphs)
