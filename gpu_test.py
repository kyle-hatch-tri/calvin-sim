print("\n\n" + "=" * 30 + " JAX GPU TEST " + "=" * 30)

import jax
from jax.lib import xla_bridge
import jax.numpy as jnp

print("jax.__version__:", jax.__version__)
print("xla_bridge.get_backend().platform:", xla_bridge.get_backend().platform)
print("jax.devices():", jax.devices())
print("jax.default_backend():", jax.default_backend())

key = jax.random.PRNGKey(42)
x = jax.random.uniform(key, (5,))
print("x:", x)
y = jnp.ones_like(x)
print("y:", y)
print("x + y:", x + y)


import jaxlib
import flax 
import optax 
import distrax 
import chex
# import orbax

print("jaxlib.__version__:", jaxlib.__version__)
print("flax.__version__:", flax.__version__)
print("optax.__version__:", optax.__version__)
print("distrax.__version__:", distrax.__version__)
print("chex.__version__:", chex.__version__)

print("=" * 70 + "\n")


print("\n\n" + "=" * 30 + " Tensorflow GPU TEST " + "=" * 30)


import tensorflow as tf


print("tf.__version__:", tf.__version__)

print("tf.test.is_gpu_available():", tf.test.is_gpu_available(), "\n")

print("tf.test.is_gpu_available(cuda_only=True):", tf.test.is_gpu_available(cuda_only=True), "\n")

print("tf.config.list_physical_devices('\GPU\'):", tf.config.list_physical_devices('GPU'), "\n")


with tf.device('gpu:0'):
    x = tf.random.uniform((5, 1))
    print("x:", x)
    y = tf.ones_like(x)
    print("y:", y)
    print("x + y:", x + y)

print("=" * 70 + "\n")


print("\n\n" + "=" * 30 + " Pytorch GPU TEST " + "=" * 30)

import torch

print("torch.__version__:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())

x = torch.rand(5).cuda()
y = torch.ones_like(x).cuda()
print("x:", x)
print("y:", y)
print("x + y:", x + y)

print("=" * 70 + "\n")




import pytorch_lightning
print("pytorch_lightning.__version__:", pytorch_lightning.__version__)

import lightning_lite
print("lightning_lite.__version__:", lightning_lite.__version__)

import torchvision
print("torchvision.__version__:", torchvision.__version__)

import torchaudio
print("torchaudio.__version__:", torchaudio.__version__)

# import horovod
# print("horovod.__version__:", horovod.__version__)

"""


    raise HorovodVersionMismatchError(name, version, installed_version) from exception
horovod.common.exceptions.HorovodVersionMismatchError: Framework pytorch installed with version 1.12.1+cu113 but found version 1.13.1+cu117.
             This can result in unexpected behavior including runtime errors.
             Reinstall Horovod using `pip install --no-cache-dir` to build with the new version.
root@TRI-251002:/opt/ml/code# p3 gpu_test.py 


Installing collected packages: mpmath, triton, sympy, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12, torch, torchvision, pytorch-lightning
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
calvin 0.0.1 requires MulticoreTSNE, which is not installed.
calvin 0.0.1 requires pyhash, which is not installed.
calvin 0.0.1 requires pytorch-lightning==1.8.6, but you have pytorch-lightning 2.1.3 which is incompatible.
calvin 0.0.1 requires torch==1.13.1, but you have torch 2.1.2 which is incompatible.
torchaudio 0.12.1+cu113 requires torch==1.12.1, but you have torch 2.1.2 which is incompatible.
torchdata 0.4.1 requires torch==1.12.1, but you have torch 2.1.2 which is incompatible.
Successfully installed mpmath-1.3.0 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.18.1 nvidia-nvjitlink-cu12-12.3.101 nvidia-nvtx-cu12-12.1.105 pytorch-lightning-2.1.3 sympy-1.12 torch-2.1.2 torchvision-0.16.2 triton-2.1.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv

"""