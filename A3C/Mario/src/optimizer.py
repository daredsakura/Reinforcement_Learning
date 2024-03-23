import torch


# 实现一个在多个进程之间共享状态的Adam优化器
class GlobalAdam(torch.optim.Adam):
    def __init__(self, params, lr):
        super(GlobalAdam, self).__init__(params=params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                #  exp_avg和exp_avg_sq是Adam算法中的核心参数,其他一些参数，如step、momentum_buffer、max_exp_avg等，会在后续的优化过程中自动更新和计算
                state['exp_avg'] = torch.zeros_like(p.data)  # 参数的梯度的指数移动平均值,创建一个与参数p的数据形状相同的全零张量
                state['exp_avg_sq'] = torch.zeros_like(p.data)  # 参数的梯度平方的指数移动平均值,创建一个与参数p的数据形状相同的全零张量
                state['exp_avg'].share_memory_()  # 使用共享内存技术
                state['exp_avg_sq'].share_memory_()
