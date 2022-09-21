import torch
from torch.utils.data import Dataset
from dgl.data.utils import load_graphs
from os.path import join
from dgl.dataloading import GraphDataLoader
import surpport.myfunction as MF


class cross_Dataset(Dataset):
    def __init__(self, arg) -> None:
        super(Dataset, self).__init__()
        self.arg = arg
        self.load(arg)

    def load(self, arg):
        f = arg.select_f  # type: str
        ori = r'E:\DATABASE\FirstGNN\CrossData'
        path = join(ori, f+'_dgl_graph')
        gdata = torch.load(path)
        self.graphs, self.labels = gdata['graphs'], gdata['labels']

    def __getitem__(self, index):
        return self.graphs[index], self.labels[index]

    def __len__(self):
        return len(self.graphs)


def c_dataload(arg, model, opt, alpha):
    """cross experiment dataload

    Args:
        arg (Args): arg
        model (nn.Moudle): your model
        opt (opt): opt
        alpha (int): less than 11 and greater than 1

    Returns:
        tuple: arg, tr_dataloader, ts_dataloader
    """
    dataset = cross_Dataset(arg)
    if not arg.new_train:
        MF.continue_tr(arg.dir, model, opt, arg)
    else:
        end_num = alpha * 1875  # 单人一个
        lenth = len(dataset)
        arg.tr_id = range(end_num)
        arg.ts_id = range(lenth-1875, lenth)

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
