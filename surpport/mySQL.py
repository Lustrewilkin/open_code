from genericpath import exists
import numpy as np
import pandas as pd
import os
from time import strftime, localtime
import json

import torch

# code snip
# timep = strftime("%Y-%m-%d")


def store_epoch(target, bt, arg, writer, epoch=1):
    """
    target必须是 np 的对象，
    名字：FirstGNN\\CT\\modelname_count\\train\\loss
    """
    timep = strftime("%Y-%m-%d")
    dir_pool = [arg.s_name, arg.model_name +
                '_{0}'.format(arg.count), bt[0], bt[1], timep]
    dir = os.path.join(arg.data_dir, arg.model_task)
    for name in dir_pool:
        dir = os.path.join(dir, name)

    if os.path.exists(dir) is False:
        os.makedirs(dir)
    # target is [-1,1] shape np.ndarray
    writer.add_scalars(
        'L&A/{0}'.format(bt[1]), {'{0}'.format(bt[0]): np.mean(target[1:])}, epoch)
    # target1 = pd.DataFrame(target[1:], columns=['epoch={0}'.format(epoch)])
    # target1.to_csv(os.path.join(dir, 'epoch_{0}.csv'.format(epoch)))


def store_all(target, discrp, arg, writer=None):
    timep = strftime("%Y-%m-%d")
    dir_pool = [arg.s_name, arg.model_name +
                '_{0}'.format(arg.count), timep]
    dir = os.path.join(arg.data_dir, arg.model_task)
    for name in dir_pool:
        dir = os.path.join(dir, name)
    tg1 = pd.DataFrame(data=target, index=[arg.s_name])


def gen_base_rcd(arg):
    # 只生成基本记录，额外的需要扩充
    # 只生成名字文件夹
    if not exists(arg.tar_path):
        os.makedirs(arg.tar_path)
    with open(arg.tar_path+'\\{0}.txt'.format(arg.discrption), 'w', encoding='utf-8') as f:
        pass

    base = {
        'time': strftime("%Y-%m-%d %a %H:%M:%S", localtime()),
        'discrption': arg.discrption,
        'task': arg.model_task,
        'model': arg.model_name,
        'count': arg.count,
        'patient': arg.s_name,
        'fre_chs': arg.select_f,
    }
    save_recorder(base, arg, 'base')
    return base


def save_recorder(data, arg, name: str):
    if exists(arg.tar_path) is False:
        os.makedirs(arg.tar_path)
    if name == None:
        with open(arg.tar_path+'\\recorder.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:
        with open(arg.tar_path+'\\{0}recorder.json'.format(name), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def save_final(epoch, model, val_acc, arg, opt):
    if (val_acc >= arg.exception_acc):
        state = {'net': model.state_dict(),
                 'opt': opt.state_dict(),
                 'epoch': epoch,
                 'tr_id': arg.tr_id,
                 'ts_id': arg.ts_id}
        torch.save(state, arg.tar_path+'\\fmodel.pth')
        print('已保存最终模型，val_acc = %.4f' % (val_acc))


def save_best(epoch, model, val_acc, arg, opt):
    if (val_acc >= arg.exception_acc) and (val_acc > arg.best_acc):
        state = {'net': model.state_dict(),
                 'opt': opt.state_dict(),
                 'epoch': epoch,
                 'tr_id': arg.tr_id,
                 'ts_id': arg.ts_id}
        arg.best_acc = val_acc
        torch.save(state, arg.tar_path+'\\bmodel.pth')
        print('已保存模型，val_acc = %.4f' % (val_acc))
    else:
        print('未达到期望，未保存模型')


def rcd_log(loss=None, acc=None, writer=None,
            recorder=None, epoch=0, stage='train'):
    """
    stage = 'train' or 'test'/'eval'
    """

    if isinstance(loss, torch.Tensor):
        loss = loss.numpy()
    if isinstance(acc, torch.Tensor):
        acc = acc.numpy()

    if stage == 'train':
        writer.add_scalar(tag='Loss/train',
                          scalar_value=np.mean(loss),
                          global_step=epoch)
        recorder['ave_tr_loss'] = float(np.mean(loss[1:], axis=0))
        recorder['tr_loss'] = np.reshape(loss, -1).tolist()[1:]

        writer.add_scalar(tag='Acc/train',
                          scalar_value=np.mean(acc),
                          global_step=epoch)
        recorder['ave_tr_acc'] = float(np.mean(acc[1:], axis=0))
        recorder['tr_acc'] = np.reshape(acc, -1).tolist()[1:]

    elif stage == 'test' or stage == 'eval':
        writer.add_scalar(tag='Loss/test',
                          scalar_value=np.mean(loss),
                          global_step=epoch)
        recorder['ave_ts_loss'] = float(np.mean(loss[1:], axis=0))
        recorder['tr_loss'] = np.reshape(loss, -1).tolist()[1:]

        writer.add_scalar(tag='Acc/test',
                          scalar_value=np.mean(acc),
                          global_step=epoch)
        recorder['ave_ts_acc'] = float(np.mean(acc[1:], axis=0))
        recorder['ts_acc'] = np.reshape(acc, -1).tolist()[1:]


def rcd_result(logits, labels, recorder=None):
    if type(logits) is torch.Tensor:
        recorder['logits'] = logits.numpy().tolist()
        recorder['labels'] = labels.numpy().tolist()
    if type(logits) is np.ndarray:
        recorder['logits'] = logits.tolist()
        recorder['labels'] = labels.tolist()
