import torch
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import os


def get_layer(model, key_list):
    for key in key_list:
        ans = model._modules[key]

    return ans


class GradCAM(object):
    def __init__(self, model, aim_layer):
        super().__init__()
        self.model = model
        self.aim_layer = get_layer(model=model, key_list=aim_layer)

        self.act = []  # 输出存储
        self.grad = []  # 梯度存储
        self.register_hooks()  # hook 注册

    def register_hooks(self):

        def forward_hook(module, input, output):  # 记录输出
            self.act.append(output.data.clone())

        def backward_hook(module, grad_input, grad_output):
            self.grad.append(grad_output[0].data.clone())

        self.aim_layer.register_forward_hook(forward_hook)
        self.aim_layer.register_backward_hook(backward_hook)

    def listclear(self):  # 保存了就清空
        self.act.clear()
        self.grad.clear()

    def gen_cam(self, g, label):
        self.listclear()

        output = self.model(g, g.ndata['h'].float(), g.edata['f'].float())
        grad_output = output.data.clone()
        grad_output.zero_()
        grad_output.scatter_(1, label.t(), 1.0)  # lable 的形状注意一下
        output.requires_grad_(True)
        grad_output.requires_grad_(True)

        # backward 默认是对最后的分类求均值再求梯度（见BP公式）
        # 如果满足其他条件，则需要输入一个向量，执行：mm(output,grad_output), 达到选择的目的
        output.backward(grad_output)
        act = self.act[0]
        grad = self.grad[0]

        # weight = grad.mean(dim = (2,3), keepdim = True)
        # cam = (weight * act).sum(dim = 1, keepdim = True)
        # cam = torch.clamp(cam,min = 0)

        cam = torch.mul(grad, act)
        cam = torch.sum(cam, dim=1)
        cam = torch.clamp(cam, min=0)
        return cam, output


def node_build(input, ctype, arg):
    input = input.numpy()
    index = input.argsort()
    inshape = input.shape[0]
    if ctype == 'size':
        size_v = 0
        rank = np.ones(inshape, dtype=np.float32)
        for i in index:
            size_v += 100
            rank[i] = size_v
            if rank[i] > 1200:
                rank[i] = 1200
        return rank

    if ctype == 'color':
        rgba_vl = [255/256, 0., 0., 0.0]  # arg.rgba
        alsort = np.ones((inshape, 4), dtype=np.float32)
        for i in index:
            rgba_vl[3] = rgba_vl[3] + (1/inshape)
            alsort[i, :] = rgba_vl
        return alsort


def show_cam(g, logits, label, cam, arg, pic_num):

    G = g.to_networkx()
    cam = torch.softmax(cam, dim=0)  # cam 是yi维的，但仍然要检查

    jet = cm = plt.get_cmap('jet')
    fig, ax = plt.subplots()

    # cam 可视化
    # color
    color_rank = node_build(cam, ctype='color', arg=arg)
    size_rank = node_build(cam, ctype='size', arg=arg)
    options = {
        "node_color": color_rank,
        "node_size": size_rank,
        # "edge_color": weights,
        # "arrows": True,
        "width": 2,
        "edge_cmap": plt.cm.Blues,
        # "node_cmap": plt.cm.YlGn,
        "with_labels": True,
        "labels": arg.povname
    }
    nx.draw(G, pos=arg.pov, **options)
    # plt.savefig('edges.png')
    plt.xlim(-1.50, 1.50)  # 设置首界面X轴坐标范围
    plt.ylim(-1.50, 1.50)  # 设置首界面Y轴坐标范围

    values, indices = logits.topk(1, dim=1, largest=True, sorted=True)
    if indices == label:
        istrue = 'True'
    else:
        istrue = 'False'
    ax.set_title("Class: {0} \nresult: {1}".format(label[0][0], istrue))

    plt.savefig(os.path.join(arg.cam_save_dir, "lable_{0}-{2}-pic_{1}.png")
                .format(label[0][0], pic_num, istrue))
    # plt.show()
