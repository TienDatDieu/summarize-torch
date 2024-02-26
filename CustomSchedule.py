import torch

class CustomSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.d_model = torch.tensor(self.d_model, dtype=torch.float32)
        self.warmup_steps = warmup_steps
        super(CustomSchedule, self).__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch + 1
        step = torch.tensor(step, dtype=torch.float32)
        arg1 = torch.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return [float(torch.rsqrt(self.d_model) * min(arg1, arg2))]