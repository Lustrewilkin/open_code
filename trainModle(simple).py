from torch.utils.tensorboard import SummaryWriter
import os
import surpport.mySQL as mySQL
import surpport.myfunction as MF
from surpport.Args import Args
from surpport.dataprocess import dataloader, my_dataset
import torch.nn.functional as F
import torch
import numpy as np
from surpport.nnstructure import Classifier2, Classifier4, Classifier6, Eq_Classifer
Classifier6

# prepare


def prepare():
    arg = Args()
    # model = Classifier4(arg)  # 实际模型
    model = Classifier6(arg)
    base_rcd = mySQL.gen_base_rcd(arg)
    recorder = {'base': base_rcd}
    return arg, model, recorder


def trainmodel(arg, model, recorder):
    if arg.device == 'cuda':
        device = torch.device('cuda:0')
        model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=6e-5,
                           eps=1e-8, weight_decay=0.1)  # weight_decay=0.6

    # dir = r'E:\DATABASE\FirstGNN\patien\wfz\GNN_10\fmodel.pth'
    dir = None
    # data
    dataset = my_dataset(name=arg.s_name, raw_dir=arg.raw_data_dir,
                         save_dir=arg.save_data_dir)
    num_labels = dataset.num_labels

    # 先分割数据集的原因是需要拼合图，后面拼合怕问题，就先随机划分比较好
    if dir is not None:
        MF.continue_tr(dir, model, opt, arg)
        train_dataset = torch.utils.data.Subset(
            dataset=dataset, indices=arg.tr_id)
        test_dataset = torch.utils.data.Subset(
            dataset=dataset, indices=arg.te_id)
    else:
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])
        arg.tr_id = train_dataset.indices
        arg.te_id = test_dataset.indices

    tr_dataloader = dataloader(
        train_dataset, arg.batch_size, collate=MF.collate, shuffle=True)
    ts_dataloader = dataloader(
        test_dataset, arg.batch_size, collate=MF.collate, shuffle=True)

    # recorder
    writer = SummaryWriter(arg.tar_path+'\\Journal')
    tr_loss_recoder = dict()
    test_loss_recoder = dict()

    # train
    for epoch in range(1, arg.epoch_num+1):
        recorder[str(epoch)+'-th'] = dict()
        rd = recorder[str(epoch)+'-th']
        # 由于图经过拼合，所以需要多一个dataloader的过程
        # 前两个是list
        tr_loss, tr_acc = MF.train(
            epoch,
            model, opt, tr_dataloader,
            arg,
            writer
        )
        mySQL.rcd_log(tr_loss, tr_acc, writer, rd, epoch, 'train')

        el_loss, el_acc, logits, labels = MF.evaluate(
            epoch,
            model, opt, ts_dataloader,
            arg,
            writer
        )

        mySQL.rcd_log(el_loss, el_acc, writer, rd, epoch, 'test')
        mySQL.rcd_result(logits, labels, rd)

        val_acc = np.mean(el_acc, axis=0)
        MF.save_best(val_acc, model, arg)

        mySQL.save_final(epoch, model, val_acc, arg, opt)

    mySQL.save_recorder(recorder, arg, 'flow')
    print('best acc: %.4f' % (arg.best_acc))
    writer.close()


name = ['kly']
f = [6]
for s_n in name:
    for f_c in f:
        Args.s_name = s_n
        Args.count = f_c
        arg, model, recorder = prepare()
        trainmodel(arg, model, recorder)
