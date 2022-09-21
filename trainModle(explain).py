import torch.nn.functional as F
import torch
import numpy as np
from surpport.nnstructure import Classifier
from surpport.dataprocess import dataloader, my_dataset
from surpport.Args import Args
import surpport.myfunction as MF
import surpport.mySQL as mySQL
from torch.utils.tensorboard import SummaryWriter
# 实例化与参数
# 这仅是个例子，特征尺寸是1

arg = Args()
device = torch.device('cuda:0')
model = Classifier(arg)
model = model.to(device)

# torch.optim.'function'(model.parameters(),lr=0.01)#用于指定优化器
opt = torch.optim.Adam(model.parameters(), lr=6e-5,
                       eps=1e-8, weight_decay=0.6)
#criterion = My_loss()

# 这里还要优化
# 数据读入
dataset = my_dataset(name=arg.s_name, raw_dir=arg.raw_data_dir,
                     save_dir=arg.save_data_dir)
#dataset = dataset.to(device)
num_labels = dataset.num_labels

# dataset_size = len(dataset)
# indices = list(range(dataset_size))
# split = int(dataset*0.75)
# train_indices, val_indices = indices[split:], indices[:split]
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# 先分割数据集的原因是需要拼合图，后面拼合怕问题，就先随机划分比较好
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])


tr_dataloader = dataloader(
    train_dataset, arg.batch_size, collate=MF.collate, shuffle=True)
ts_dataloader = dataloader(
    test_dataset, arg.batch_size, collate=MF.collate, shuffle=True)
# cam_dataloader = dataloader(test_dataset, batch_size=1, collate=MF.collate)
# 确认运行设备
# print([model.device , dataloader.device])

# 记录loss
writer = SummaryWriter('E:\\CODEBASE\\myDGL\\FirstDGL\\{2}-Journal\\{0}_{1}'.format(
    arg.model_name, arg.count, arg.model_task))
tr_loss_recoder = []
test_loss_recoder = []
# lr 迭代
#scheduler = opt.lr_scheduler.CosineAnnealingLR(opt, T_max = arg.T_MAX)


# pre_process(dataset)
cam_shift = 0
for epoch in range(1, arg.epoch_num+1):
    # 由于图经过拼合，所以需要多一个dataloader的过程
    loss_list, acc_list = MF.train(
        epoch,
        model, opt, tr_dataloader,
        arg,
        writer
    )

    mySQL.store(loss_list, bt=['train', 'loss'],
                arg=arg, writer=writer, epoch=epoch)
    mySQL.store(acc_list, bt=['train', 'acc'],
                arg=arg, writer=writer, epoch=epoch)

    loss_list, acc_list, val_loss, val_acc = MF.evaluate(
        epoch,
        model, opt, ts_dataloader,
        arg,
        writer
    )
    mySQL.store(loss_list, bt=['test', 'loss'],
                arg=arg, writer=writer, epoch=epoch)
    mySQL.store(acc_list, bt=['test', 'acc'],
                arg=arg, writer=writer, epoch=epoch)
    # if val_acc > 0.78 and cam_shift == 0:
    #     MF.CAM(model, epoch, cam_dataloader, arg, writer)
    #     cam_shift = 1

    MF.save_best(val_acc,
                 epoch,
                 model, opt, dataloader,
                 arg
                 )
    MF.save_final(epoch, model, val_acc, arg)

print('best acc: %.4f' % (arg.best_acc))
writer.close()
