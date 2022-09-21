from torch.autograd import grad
import torch.nn.functional as F
import torch
import numpy as np
import dgl
import matplotlib.pyplot as plt
from surpport.Args import Args
import surpport.gradCAM as gradCAM
import os
# import gradCAM
# 对图进行拼合


def collate(samples):
    # 用map提取了samples的每个元素，在list里拼合再tensor，但应该是可以直接的
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    # 复杂的原因是特征只有一维，用迭代器才可以array
    batched_labels = torch.tensor(
        np.array([item.detach().numpy() for item in labels]))
    #batched_labels = torch.tensor(lables)
    return batched_graph, batched_labels
# ————————————————
# 版权声明：本文为CSDN博主「cqu_shuai」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/beilizhang/article/details/111936390


def convert_oh(output, target):
    # target 需要一个数字tensor
    oh = torch.zeros_like(output.shape[1])
    oh = oh.scatter(dim=1, index=target, value=1)
    return oh


def loss_fuction(output, target):

    loss = F.cross_entropy(
        output, target.long().view(-1).cuda())
    # b_n = torch.zeros_like(output)
    # b_n = b_n.scatter(dim=1,index = target,value = 1)
    # loss = F.mse_loss(output,b_n)
    return loss


def acc_function(output, target):
    # dim = 1 求行的softmax
    # """
    #     KL散度
    # """
    # a_n = F.softmax(output,dim = 1)
    # b_n = torch.zeros_like(a_n)
    # b_n = b_n.scatter(dim=1,index = target,value = 1)
    # acc = 1-F.kl_div(a_n,b_n).abs() #后期再改
    """
        Hellinger distance    
    """
    # a_n = F.softmax(output,dim = 1)
    # b_n = torch.zeros_like(a_n)
    # b_n = b_n.scatter(dim=1,index = target,value = 1)
    # res = cha.mean()*cha.shape[0]*cha.shape[1]
    # acc = (1/torch.sqrt(torch.tensor(2))*torch.sqrt(res))

    values, indices = output.topk(1, dim=1, largest=True, sorted=True)
    acc = (indices.cuda() == target.cuda()).float().mean()
    return acc


def display_eval(epoch, iter_1,
                 arg,
                 loss_list, acc_list):
    if iter_1 % arg.display_freq == 0:
        print("Epoch %02d, Iter %03d,"
              "train loss = %.4f, train acc = %.4f"
              % (epoch, iter_1,
                 torch.mean(loss_list),
                 torch.mean(acc_list)))


def ROC_display(output, target):
    TP, FP, TN, FN = 0., 0., 0., 0.
    # values, indices = output.topk(1, dim=1, largest=True, sorted=True)
    # 计算
    ix, iy = 0, 0
    x_l, y_l = [], []
    mp = (torch.ones_like(target).t()) * target
    mn = len(output) - mp
    for i in range(len(output)):
        if output[i, 0] > output[i, 1]:
            iy = iy + 1/mp
            x_l.append(ix)
            y_l.append(iy)
        else:
            ix = ix + 1/mn
            x_l.append(ix)
            y_l.append(iy)
    plt.plot(x_l, y_l)
    plt.show()


def train(epoch,
          model, opt, dataloader,
          arg,
          writer=None, recorder=None):
    loss_list = torch.tensor([]).cuda()
    acc_list = torch.tensor([]).cuda()
    iter_1 = 0
    st = 0

    # 记录到底有一个epoch有几次iter
    #len_data = dataloader.__len__

    model.train()

    # 会自动跑完整个训练集
    for g, labels in dataloader:
        # 特征提取
        iter_1 = iter_1+1
        if arg.device == 'cuda':
            cuda_g = g.to(torch.device('cuda:0'))

        logits, _ = model(cuda_g)
        loss = loss_fuction(logits, labels)  # tensor
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc = acc_function(logits, labels)  # tensor

        # 记录
        loss_list = torch.cat([loss_list, loss.detach().view(-1)])
        acc_list = torch.cat([acc_list, acc.detach().view(-1)])

        # 展示输出
        display_eval(epoch, iter_1, arg, loss, acc)
    loss_list = loss_list.cpu().numpy()
    acc_list = acc_list.cpu().numpy()
    return loss_list, acc_list


def evaluate(epoch,
             model, opt, dataloader,
             arg,
             writer=None, recorder=None):
    val_loss = 0
    val_acc = 0
    loss_list = torch.tensor([]).cuda()
    acc_list = torch.tensor([]).cuda()
    logits_list = torch.zeros((1, 5)).cuda()
    labels_list = torch.zeros((1, 1)) # only labels not in GPU
    model.eval()

    with torch.no_grad():
        iter_1 = 0

        for g, labels in dataloader:

            cuda_g = g.to('cuda:0')
            logits, _ = model(cuda_g)

            loss = loss_fuction(logits, labels)
            acc = acc_function(logits, labels)

            val_loss += loss.item()
            val_acc += acc.item()

            # 记录
            loss_list = torch.cat([loss_list, loss.detach().view(-1)])
            acc_list = torch.cat([acc_list, acc.detach().view(-1)])
            logits_list = torch.cat([logits_list, logits.detach().view(-1, 5)])
            labels_list = torch.cat([labels_list, labels.detach().view(-1, 1)])
            iter_1 += 1

            # if writer != None:
            #     alf = len(dataloader)
            #     writer.add_scalars(
            #         'L&A/Loss', {'test': torch.mean(loss)}, (epoch-1)*alf+iter_1)
            #     writer.add_scalars('L&A/Accurance', {'test': torch.mean(acc)},
            #                        (epoch-1)*alf+iter_1)

    val_loss = val_loss/iter_1
    val_acc = val_acc/iter_1

    # 仅限 2 分类, 多分类还没写
    # ROC_display(logits_recorder,lables_recorder)
    print('Epoch %02d, val loss = %.4f, val acc = %.4f'
          % (epoch, val_loss, val_acc))
    loss_list = loss_list .cpu().numpy()
    acc_list = acc_list.cpu().numpy()
    logits_list = logits_list.cpu().numpy()
    labels_list = labels_list.cpu().numpy()

    return loss_list[1:], acc_list[1:], logits_list, labels_list


def save_best(val_acc, model, arg):
    if val_acc >= arg.exception_acc:
        if val_acc > arg.best_acc:
            arg.best_acc = val_acc
            torch.save(model.state_dict(), arg.tar_path+'\\bmodel.pth')
            print('已保存模型，best_acc = %.4f' % (arg.best_acc))
    else:
        print('未达到期望，未保存模型')


def CAM(model, epoch, dataloader, arg):
    model.eval()
    pic_num = 0
    expalianer = gradCAM.GradCAM(model=model, aim_layer=arg.aim_layer)
    for g, label in dataloader:
        cam, logits = expalianer.gen_cam(g, label)
        gradCAM.show_cam(g, logits=logits, label=label,
                         cam=cam, arg=arg, pic_num=pic_num)
        pic_num += 1
    return 0


def continue_tr(dir, model, opt, arg):
    checkpoint = torch.load(dir)
    model.load_state_dict(checkpoint['net'])
    opt.load_state_dict(checkpoint['optimizer'])
    arg.tr_id = checkpoint['tr_id']
    arg.te_id = checkpoint['te_id']
    # start_epoch = checkpoint['epoch'] + 1
    print('已加载相关参数：model, opt, sub_dataSet')


def recSch_lr(optimizer, epoch, acc, lr):
    # 效果不好的原因是没有继承参数，应用opt的para group 进行调整
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    if acc < 0.7 and acc > 0.6:
        lr_r = lr * 0.5
    elif acc > 0.85:
        lr_r = lr * 0.25
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_r


def keepValue(v_list, v):
    v = np.reshape(torch.clone(v).detach().cpu().numpy(), 1)
    v_list = np.concatenate([v_list, v], axis=0)


def mode_device(s: str, model):
    """select model device

    Args:
        s (str): 'cuda' or 'cpu' are avialabel
        model (nn.Moudle): your model
    """
    if s == 'cuda':
        device = torch.device('cuda:0')
        model = model.to(device)
