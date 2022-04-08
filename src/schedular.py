import math


class CosineSchedularLinearWarmup:
    def __init__(self, optimizer, steps_per_epoch, warmup_epochs, epochs, lr):
        self.opt = optimizer
        self.total_steps = steps_per_epoch * epochs
        self.wamrup_steps = warmup_epochs * steps_per_epoch
        self.decay_steps = self.total_steps - self.wamrup_steps
        self.step = 0
        self.lr = lr

    def get_scale(self):
        if self.wamrup_steps > self.step:
            return self.step / self.wamrup_steps
        else:
            ratio = (self.step - self.wamrup_steps) / self.decay_steps
            return 0.5 * (1 + math.cos(ratio * math.pi))

    def step(self):
        scale = self.get_scale()
        for param_group in self.opt.param_groups:
            param_group["lr"] = self.lr * scale
            self.step += 1
            return param_group["lr"]
