from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Create a new custom learning rate scheduler that inherits from LR scheduler
    """

    def __init__(self, optimizer, step_size, gamma, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        self.step_size = step_size
        self.gamma = gamma
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Get lr scheduler, output a list
        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        if self.last_epoch % self.step_size == 0 and self.last_epoch > 0:
            return [base_lr * self.gamma for base_lr in self.base_lrs]
        else:
            return [base_lr for base_lr in self.base_lrs]
