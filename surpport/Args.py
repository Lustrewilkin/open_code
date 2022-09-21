import numpy as np
import os
import time


class Args():
    def __init__(self):
        """after needing to eval func: m_info
        additions: m_xxxx is model prefix, 
        """
        now = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
        self.discrption = f'model run at {now}'
        self.m_Hyperparameter()
        self.m_paremeters()
        self.expariment_info()
        self.presatation()
        self.end_point()

    # // ! 只能在这里面改，因为DGLdataset的init限制了
    # 修复了
    def d_prepare(self, p_name='dzq', freq=2, ):
        """this func just prepares for main data. For other data flow, 
        u should overwriter it.

        Args:
            p_name (str): people name, more info read self.patient
            freq (int): for our data-preprocessing, graph structure should 
                relate frequency
        """
        # 数据描述
        self.s_name = p_name
        self.fre_chs = freq  # 频率点选择
        self.select_f = 'f'+str(self.fre_chs)

    def m_info(self, m_name, m_task, num, device='cuda'):
        """model infomation readin

        Args:
            m_name (str): model name, if you didn't rectify or build other 
                networks, u should keep his name.
            m_task (str): there are many tasks in model estimation, for your
                works and experiment, you better do it.
            num (int): for ur special task get a num to remark it, and note 
                what's happend or the result.
            device (str): it's 'cuda' or 'cpu'
        """
        # 模型信息
        self.new_train = True
        self.model_name = m_name
        self.model_task = m_task
        self.count = num
        self.file_name = self.model_task + '-' + \
            self.model_name + '-' + str(self.count)
        # 设备
        self.device = 'cuda'  # or 'cuda'
        self.data_path()

    def m_Hyperparameter(self):
        # 超参数
        self.batch_size = 32  # 大一些计算快
        self.epoch_num = 160  # 不需要那么多, 200 差不多
        self.pip_num = 3
        self.dr = 0.2
        T_MAX = 1  # 不要大于epoch——num
        # https://zhuanlan.zhihu.com/p/261134624?utm_source=wechat_session

    def m_paremeters(self):
        """it's static func, if u want to rec, please rec in script.
        """
    # 输入参数
        self.input_ft = 128
        self.num_labels = 5

    # GRAPH 存读路径
    def data_path(self):
        data_dir = 'E:\\DATABASE\\FirstGNN'
        self.raw_data_dir = data_dir + '\\'+'signal'
        self.save_data_dir = data_dir + '\\'+'graphbin'
        self.save_modle_path = data_dir + '\\'+'modelset'
        # cam_save_dir = data_dir+'\\'+'cam'
        # record_dir = data_dir+'\\'+'recorder'
        self.tar_path = os.path.join(
            data_dir, *[str(self.model_task), str(self.s_name), self.model_name+'_'+str(self.count)])

    def presatation(self):
        # 显示
        self.display_freq = 5  # 不要超过epoch

    # save model
    def end_point(self):
        self.save_frq = 2
        self.exception_acc = 0.90
        self.best_acc = 0.0

    def expariment_info(self):

        self.patient = ['dzq', 'kly', 'lbg', 'llw', 'my',
                        'qeq', 'sll', 'swl', 'wfz', 'xjc', 'yc', 'ynb']
        self.task = ['double01', 'double02',
                     'double03', 'simple01', 'simple02']

        self.povname = {0: 'P3', 1: 'C3', 2: 'F3', 3: 'Fz', 4: 'F4', 5: 'C4', 6: 'P4', 7: 'Cz', 8: 'Fp1', 9: 'Fp2', 10: 'T3', 11: 'T5',
                        12: 'O1', 13: 'O2', 14: 'F7', 15: 'F8', 16: 'T6', 17: 'T4'}

        self.pov = {0	:	np.	array(	[-0.545,	-0.67302	]	)	,
                    1	:	np.	array(	[-0.71935,	0.]	)	,
                    2	:	np.	array(	[-0.545,	0.67302	]	)	,
                    3	:	np.	array(	[	0.,	0.71935	]	)	,
                    4	:	np.	array(	[	0.545,	0.67302	]	)	,
                    5	:	np.	array(	[	0.71935,	0.	]	)	,
                    6	:	np.	array(	[	0.545,	-0.67302	]	)	,
                    7	:	np.	array(	[	0.,	0.	]	)	,
                    8	:	np.	array(	[-0.30883,	0.95048	]	)	,
                    9	:	np.	array(	[	0.30883,	0.95048	]	)	,
                    10	:	np.	array(	[-0.99457,	0.	]	)	,
                    11	:	np.	array(	[	0.80498,	-0.58672	]	)	,
                    12	:	np.	array(	[-0.30883,	-0.95048	]	)	,
                    13	:	np.	array(	[	0.30883,	-0.95048	]	)	,
                    14	:	np.	array(	[-0.80852,	0.58743	]	)	,
                    15	:	np.	array(	[	0.80852,	0.58743	]	)	,
                    16	:	np.	array(	[	0.80498,	-0.58672	]	)	,
                    17	:	np.	array(	[	0.99457,	0.	]	)
                    }
