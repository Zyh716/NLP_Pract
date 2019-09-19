import numpy as np


class ScheduledOptim():
    '''三个功能，获得学习率、更新学习率、两者都做'''
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self.optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps
        ])

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_lr(self):
        self.n_current_steps += 1
        lr = self.init_lr * self.get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def step_and_update_lr(self):
        self.update_learning_rate()
        self.optimizer.step()