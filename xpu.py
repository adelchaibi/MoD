import torch 
from typing import Any, Dict, List, Optional, Union
from pytorch_lightning import Trainer


import pytorch_lightning as pl
from pytorch_lightning.accelerators.accelerator import Accelerator


import intel_extension_for_pytorch

class XPUAccelerator(Accelerator):
    """Experimental support for XPU, optimized for large-scale machine learning."""

    @staticmethod
    def parse_devices(devices):
        # Put parsing logic here how devices can be passed into the Trainer
        # via the `devices` argument
        print("===========",devices)
        return [devices]

    @staticmethod
    def get_parallel_devices(devices):
        # Here, convert the device indices to actual device objects
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        # Return a value for auto-device selection when `Trainer(devices="auto")`
        print("auto_device_count")
        return 1
        #return xpulib.available_devices()

    @staticmethod
    def is_available() -> bool:
        print("is_available")
        return True
        #return xpulib.is_available()

    def get_device_stats(self, device: Union[str, torch.device]):
        # Return optional device statistics for loggers
        print("get_device_state")
        return torch.xpu.memory_stats(device)



if __name__ == '__main__':
    accelerator = XPUAccelerator()
#    trainer = Trainer(accelerator=accelerator, devices=1)

