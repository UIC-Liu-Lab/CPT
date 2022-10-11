"""Modify huggingface Accelerator for CPT system."""
import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer


class myAccleratedOptimizer(AcceleratedOptimizer):
    def __init__(self, optimizer, device_placement=True, scaler=None):
        super(myAccleratedOptimizer, self).__init__(
            optimizer, device_placement, scaler)

    def my_step(self, type, t, mask_back):
        self.optimizer.my_step(type='mask', t=t, mask_back=mask_back)


class myAccelerator(Accelerator):
    def __init__(
            self, device_placement: bool = True, split_batches: bool = False, fp16: bool = None, cpu: bool = False
    ):
        super(myAccelerator, self).__init__(
            device_placement, split_batches, fp16, cpu)

    def my_prepare_optimizer(self, optimizer):
        return myAccleratedOptimizer(optimizer, device_placement=self.device_placement, scaler=self.scaler)

    def my_prepare_one(self, obj):
        if isinstance(obj, torch.utils.data.DataLoader):
            return self.prepare_data_loader(obj)
        elif isinstance(obj, torch.nn.Module):
            return self.prepare_model(obj)
        elif isinstance(obj, torch.optim.Optimizer):
            optimizer = self.my_prepare_optimizer(obj)
            self._optimizers.append(optimizer)
            return optimizer
        else:
            return obj

    def my_prepare(self, *args):

        result = tuple(self.my_prepare_one(obj) for obj in args)

        return result
