import numpy as np

import torch.nn as nn


class ScheduleAdam():

    def __init__(self, optimizer, lr, warm_steps):
        # self.init_lr = np.power(hidden_dim, -0.5) # 1 / sqrt(hidden_dim)
        self.init_lr = lr
        self.optimizer = optimizer
        self.current_steps = 0
        self.warm_steps = warm_steps

    def step(self):
        # current_step 정보를 이용해서 lr Update
        self.current_steps += 1
        lr = self.init_lr * self.get_scale()

        for p in self.optimizer.param_groups:
            p['lr'] = lr # lr Update

        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_scale(self):
        return np.min([np.power(self.current_steps, -0.5),
                       self.current_steps * np.power(self.warm_steps, -1.5)
                       ])

class LambdaLR:
    def __init__(self, epochs, offset, decay_start_epoch):
        assert (epochs - decay_start_epoch) > 0, "전체 epoch가 decay_start_epoch보다 커야함"

        self.num_epochs = epochs # 설정한 총 epoch
        self.offset = offset # (저장했었던) start epoch
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch): # epoch : 현재 epoch
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.num_epochs - self.decay_start_epoch)
