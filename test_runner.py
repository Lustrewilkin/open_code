import time
from torch.utils.tensorboard import SummaryWriter
import os
import surpport.mySQL as mySQL
import surpport.myfunction as MF
from surpport.Args import Args
from surpport.dataprocess import dataload
import torch.nn.functional as F
import torch
import numpy as np
from surpport.nnstructure import Classifier7
from Experiment.Baseline.NoneGraph.EEGNet import EEGNet2
from Experiment.Baseline.GraphModel.GNN.GNN import GCN, DGCN


def m_pre():
    arg = Args()
    # model = Classifier7(arg)
    model = EEGNet2()
    model = model.cuda()
    return arg, model


def recorder_build(arg):
    base_rcd = mySQL.gen_base_rcd(arg)
    recorder = {'base': base_rcd}
    return recorder


def train(arg, model, recorder, opt, tr_dataloader, ts_dataloader, writer):
    # train
    for epoch in range(1, arg.epoch_num+1):
        # print(epoch)
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
    return None


start_time = time.perf_counter()
# names = ['dzq', 'kly', 'lbg', 'llw', 'my', 'qeq',
#  'sll', 'swl', 'wfz', 'xjc', 'yc', 'ynb']
# fs = [2, 6, 7, 15]
# head = [1, 2, 3, 4, 5, 6, 7, 8, 9]
fs = [2]
names = ['kly']
dec_base = [0.00001, 0.00003]
w_dec = []
for i in range(13):
    w_dec.append(dec_base[i % 2] * 10**(i//2))

count = 0

for name in names:
    for wd in w_dec:
        arg = Args()
        arg.pip_num = 3
        arg.dr = 0
        model = Classifier7(arg)
        model = model.cuda()
        arg.d_prepare('kly', 2)
        # ! 每次运行都改
        count += 1
        arg.m_info(m_name='m7', m_task='220831_exp', num=77,)
        base_rcd = mySQL.gen_base_rcd(arg)
        recorder = {'base': base_rcd}
        writer = SummaryWriter(arg.tar_path+'\\Journal')

        opt_arg = {'params': model.parameters(), 'weight_decay': wd}
        #    'lr': 6e-5, 'eps': 1e-8, 'weight_decay': 0.1}
        opt = torch.optim.Adam(**opt_arg)

        arg, tr_dataloader, ts_dataloader = dataload(arg, model, opt)

        train(arg, model, recorder, opt, tr_dataloader, ts_dataloader, writer)
        writer.close()

end_time = time.perf_counter()
print(f'cost: {(start_time-end_time)*1e6}s')
