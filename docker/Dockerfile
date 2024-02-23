# -------------------------
# Base SageMaker image
# -------------------------
# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker
# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker
# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker
# See https://github.com/aws/deep-learning-containers/blob/master/available_images.md

ENV PROJECT=calvin-sim

# https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425
SHELL ["/bin/bash", "-ex", "-o", "pipefail", "-c"]




#=============================
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTORCH_VERSION=1.12.1+cu113
ENV TORCHVISION_VERSION=0.13.1+cu113
ENV TORCHAUDIO_VERSION=0.12.1

ENV CUDNN_VERSION=8.2.1.32-1+cuda11.3
ENV NCCL_VERSION=2.9.9-1+cuda11.3
ENV DEBIAN_FRONTEND=noninteractive

# Install basic libraries
RUN apt-get update && apt-get install -y \
    build-essential cmake git curl docker.io vim wget ca-certificates

# Install python and pip
RUN apt-get install -y python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

# Install pytorch
RUN pip install  \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --extra-index-url https://download.pytorch.org/whl/cu113

# TODO UNCOMMENT
# Install CUDNN and NCCL
RUN apt-get install -y \
    libcudnn8=${CUDNN_VERSION} \
    libnccl2=${NCCL_VERSION}

# Install extra packages (apt-get)
RUN apt-get install -y \
    ffmpeg \
    tmux

# UNCOMMENT
# Install extra packages (pip)
RUN pip install                 \
    tqdm==4.61.0                \
    boto3==1.17.83              \
    termcolor==1.1.0            \
    pyyaml==5.4.1               \
    wandb==0.10.31              \
    opencv-python==4.5.2.52     \
    flow_vis==0.1               \
    matplotlib==3.3.4           \
    fire==0.4.0                 \
    pyquaternion==0.9.9         \
    pandas==1.1.5               \
    xarray==0.16.2              \
    diskcache==5.2.1            \
    tenacity==7.0.0             \
    awscli==1.19.101            \
    ref==0.0.2.2                \
    positional-encodings==4.0.0 \
    einops==0.3.2               \
    wget==3.2                   \
    kornia==0.6.8               \
    lpips==0.1.4                \
    scikit-image==0.19.3        \
    timm==0.6.11

# Install CamViz dependencies
RUN pip install                 \
    pygame==2.0.1               \
    PyOpenGL==3.1.5             \
    PyOpenGL-accelerate==3.1.5
RUN apt-get install -y          \
    mesa-utils                  \
    freeglut3-dev               \
    libsdl2-2.0-0               \
    python-pygame

# Install CLIP
RUN pip install ftfy regex
RUN pip install git+https://github.com/openai/CLIP.git
# Install torch-scatter
RUN pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
# # Install PyTorch3D
# RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# # Install LieTorch
# RUN pip install git+https://github.com/princeton-vl/lietorch.git@91887c3be4fe72aefe4af1deb67f2ea9ce17a049
# # Install nuScenes devkit
RUN pip install nuscenes-devkit==1.1.7
# Upgrade protobuf
RUN pip install --upgrade protobuf==3.20.0
# Install transformer stuff
RUN pip install sacremoses huggingface_hub transformers
# Install open clip
RUN pip install open_clip_torch
# # Install more transformers stuff
# RUN pip install mmcv taming-transformers taming-transformers-rom1504
# Install seaborn for extrinsic error map visualization
RUN pip install seaborn==0.12.2
#===========================



# Create an environment setup file separate from /etc/bash.bashrc, and source
# this file from /etc/bash.bashrc. The reason for this is that /etc/bash.bashrc
# doesn't execute when not running the shell interactively. If we have a separate
# file for environment setup, then we can source that file when executing
# non-interactively and still have the expected environment setup.
# This is useful for e.g., running tests with the expected environment.
# Takeaway: Instead of adding stuff to /etc/bash.bashrc, add it to this file!
ENV SHELL_SETUP_FILE=/usr/local/bin/${PROJECT}_env_setup.sh
RUN touch ${SHELL_SETUP_FILE} \
    && chmod +x ${SHELL_SETUP_FILE} \
    && echo "source ${SHELL_SETUP_FILE}" >> /etc/bash.bashrc

# -------------------------
# System tools
# -------------------------
# Configure apt for use in Docker.
RUN echo 'APT::Get::Assume-Yes "true";' | tee /etc/apt/apt.conf.d/90yes
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
    && apt-get install --no-install-recommends -q -y \
        locales \
        openssh-client openssh-server \
        lsb-release \
        cmake \
        curl \
        docker.io \
        ffmpeg \
        git \
        htop \
        libsm6 \
        libxext6 \
        libgl1  \
        libglib2.0-0 \
        libxrender-dev \
        libxext6 \
        libosmesa6-dev \
        libglfw3 \
        libjpeg-dev \
        libpng-dev \
        patchelf \
        python3-flake8-docstrings \
        python3-pip \
        python3-pyaudio \
        python3-pytest-cov \
        sudo \
        freeglut3-dev \
        software-properties-common \
        tmux \
        libasound2-dev \
    && rm -rf /var/lib/apt/lists/*
RUN locale-gen en_US.UTF-8
ENV \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# # -------------------------
# # Install ROS
# # -------------------------
# ENV ROS_PYTHON_VERSION=3
# # Add ros universe as source and install ros-dev-tools.
# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
#     curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
# RUN add-apt-repository universe
# RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg
# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
# RUN apt-get update -y \
#     && apt-get install --no-install-recommends -q -y \
#     ros-dev-tools \
#     && rm -rf /var/lib/apt/lists/*
# # Install ros2 python dependencies.
# RUN python3 -m pip install -U \
#     colcon-common-extensions \
#     flake8-blind-except \
#     flake8-builtins \
#     flake8-class-newline \
#     flake8-comprehensions \
#     flake8-deprecated \
#     flake8-import-order \
#     flake8-quotes \
#     "pytest>=5.3" \
#     pytest-repeat \
#     pytest-rerunfailures \
#     catkin_pkg \
#     empy \
#     lark
# # Build ros2 humble from source because humble doesn't distribute ubuntu20.04 binaries.
# RUN mkdir -p /opt/ros/humble/src
# WORKDIR /opt/ros/humble
# RUN vcs import --input https://raw.githubusercontent.com/ros2/ros2/humble/ros2.repos src
# RUN apt-get update -y \
#     && rosdep init \
#     && rosdep update \
#     && rosdep install --from-paths src --ignore-src -y --skip-keys "fastcdr rti-connext-dds-6.0.1 urdfdom_headers" \
#     && rm -rf /var/lib/apt/lists/*
# # Specify the correct python version here otherwise the build uses a mix of versions.
# RUN /opt/conda/bin/python3.9 -m colcon build --symlink-install --packages-skip qt_gui_cpp --packages-skip-by-dep qt_gui_cpp
# RUN echo "source /opt/ros/humble/install/local_setup.bash" >> ${SHELL_SETUP_FILE}

# # Install cv_bridge independently.
# WORKDIR /ros2_ws/src
# RUN git clone https://github.com/ros-perception/vision_opencv.git
# WORKDIR /ros2_ws
# RUN /bin/bash -c "source /opt/ros/humble/install/local_setup.bash; colcon build"
# RUN echo "source /ros2_ws/install/local_setup.bash" >> ${SHELL_SETUP_FILE}

# # -------------------------
# # Install ZED2 SDK
# # -------------------------
# RUN wget https://download.stereolabs.com/zedsdk/4.0/cu118/ubuntu20 -O ZED_SDK_Ubuntu20_cuda11.8_v4.0.2.zstd.run
# RUN chmod +x ./ZED_SDK_Ubuntu20_cuda11.8_v4.0.2.zstd.run && ./ZED_SDK_Ubuntu20_cuda11.8_v4.0.2.zstd.run -- silent
# RUN rm ZED_SDK_Ubuntu20_cuda11.8_v4.0.2.zstd.run
# ENV MESA_GL_VERSION_OVERRIDE 3.3

# -------------------------
# Python packages
# -------------------------
RUN pip install --upgrade pip

RUN pip install wheel cmake==3.18.4



# Install basic libraries
RUN apt-get update && apt-get install -y \
    build-essential cmake git curl docker.io vim wget ca-certificates

# # Install CamViz dependencies
# RUN pip install  PyOpenGL==3.1.5
# RUN pip install  PyOpenGL-accelerate==3.1.5
# RUN apt-get install -y mesa-utils
# RUN apt-get install -y freeglut3-dev
# RUN apt-get install -y libsdl2-2.0-0


# # Install CamViz dependencies
# RUN pip install                 \
#     pygame==2.0.1               \
#     PyOpenGL==3.1.5             \
#     PyOpenGL-accelerate==3.1.5
# RUN apt-get install -y          \
#     mesa-utils                  \
#     freeglut3-dev               \
#     libsdl2-2.0-0               \
#     python-pygame


# standard deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN pip install wheel cmake==3.18.4

# RUN ./install.sh

# RUN cd calvin_env/tacto
# RUN pip install -e .
# RUN cd ..
# RUN pip install -e .
# RUN cd ../calvin_models
# RUN pip install -e .


### UNCOMMENT
# # Install cython==3.0.0 after other requirements. This conflicts with
# # the requirements of robosuite, and breaks some functionality of it,
# # but otherwise pip cannot solve for a configuration.
# RUN pip install cython==3.0.0
# # Install AWS CLI v2 (and uninstall v1). You need v2 for sagemaker sso login.
# RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
#     unzip awscliv2.zip && \
#     ./aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update && \
#     rm awscliv2.zip && \
#     pip uninstall -y awscli && \
#     aws configure set default.s3.max_concurrent_requests 100 && \
#     aws configure set default.s3.max_queue_size 10000

# # ----- put additional custom deps here -----
# RUN pip install open_clip_torch==2.16.0
# RUN export FORCE_CUDA=1 && pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# RUN pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.13.1_cu117.html
# RUN pip install xformers==0.0.16

# # -------------------------
# # Install kaolin-wisp
# # -------------------------
# RUN cd /tmp && git clone https://github.com/NVIDIAGameWorks/kaolin-wisp.git
# RUN cd /tmp/kaolin-wisp && git checkout cb47e10 && export FORCE_CUDA=1 && python setup.py develop

# -------------------------
# Project code
# -------------------------
ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
ENV WORKSPACE=/opt/ml/code
COPY . ${WORKSPACE}




WORKDIR ${WORKSPACE}
ENV PYTHONPATH="${WORKSPACE}:$PYTHONPATH"
RUN git config --global --add safe.directory ${WORKSPACE}

# # -------------------------
# # Install ffo
# # -------------------------
# RUN cd /opt/ml/code/external/ffo/vlm/models/grounded_sam/ && pip install -e segment_anything
# RUN cd /tmp && git clone https://github.com/IDEA-Research/GroundingDINO.git
# RUN cd /tmp/GroundingDINO && git checkout 498048b && export FORCE_CUDA=1 && pip install -e .

# # -------------------------
# # Install CoppeliaSim
# # -------------------------
# RUN cd /tmp && \
#     wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz && \
#     tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
# ENV COPPELIASIM_ROOT=/tmp/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
# ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# #--------------------------
# # Install PyRep
# # -------------------------
# # We use pip to compile the cffi files in installing PyRep. Note that we do
# # not run `pip install -r requirements.txt`, since cffi is in conflict with
# # the diffusion policy.
# RUN cd /tmp && \
#     git clone https://github.com/stepjam/PyRep.git && \
#     cd PyRep && \
#     pip install --no-deps . 

# Add external packages to python path.
ENV EXTERNAL="${WORKSPACE}/external"
# ENV PYTHONPATH="${EXTERNAL}/diffusion_policy:$PYTHONPATH"
# ENV PYTHONPATH="${EXTERNAL}/datasets:$PYTHONPATH"
# ENV PYTHONPATH="${EXTERNAL}/datasets/externals/camviz:$PYTHONPATH"
# ENV PYTHONPATH="${EXTERNAL}/vidar:$PYTHONPATH"
# ENV PYTHONPATH="${EXTERNAL}/voltron:$PYTHONPATH"
# ENV PYTHONPATH="${EXTERNAL}/ffo:$PYTHONPATH"
# ENV PYTHONPATH="${EXTERNAL}/RLBench:$PYTHONPATH"
# ENV PYTHONPATH="${EXTERNAL}/dlimp:$PYTHONPATH"


# COPY ../susie $EXTERNAL
ENV PYTHONPATH="${EXTERNAL}/susie:$PYTHONPATH"
ENV PYTHONPATH="${EXTERNAL}/jaxrl_m:$PYTHONPATH"

ENV PYTHONPATH="${WORKSPACE}/calvin_models:$PYTHONPATH"
ENV PYTHONPATH="${WORKSPACE}/calvin_env:$PYTHONPATH"
ENV PYTHONPATH="${WORKSPACE}/calvin_env/tacto:$PYTHONPATH"
ENV PYTHONPATH="${EXTERNAL}/urdfpy:$PYTHONPATH"
ENV PYTHONPATH="${EXTERNAL}/networkx:$PYTHONPATH"

# # Instead of manually adding the robot_interface to the PYTHONPATH
# # we use pip in order to compile the protobuf files. Note the `--no-deps`
# # command, which avoids overwriting previously-installed dependencies.
# # Because we take this approach, the code referenced when importing
# # this package is not under external/robot_interface, but rather where
# # pip installs packages (we don't use editable mode b/c it doesn't
# # compile the protobufs for some reason). This repo is added as a git
# # submodule as opposed to being installed from github directly in order
# # to avoid storing gredentials in the docker image.
# RUN pip install --no-deps ${EXTERNAL}/robot_interface

# # Compile the ros2 robot policy interface.
# WORKDIR /ros2_ws
# COPY ../external/ros2_robot_policy_interface src/ros2_robot_policy_interface
# RUN /bin/bash -c "source /opt/ros/humble/install/local_setup.bash; colcon build --packages-select ros2_robot_policy_interface"
# WORKDIR WORKSPACE

# RUN cd ./external/dlimp 
# RUN pip install -e .
# RUN cd ../..

# # Need to do these separately for some reason
# RUN pip uninstall -y tensorflow
# RUN pip install tensorflow[and-cuda] 
# RUN pip install tensorflow_datasets>=4.9.2 
# RUN pip install tqdm-multiprocess==0.0.11 
# RUN pip install pre-commit==3.3.3
# RUN pip install typing_extensions

# RUN pip install --upgrade "jax[cuda11_pip]"==0.4.11 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
#     jaxlib==0.4.11 \
#     flax==0.7.0 \
#     optax==0.1.5 \
#     distrax==0.1.2 \
#     chex==0.1.7 
RUN pip install --upgrade "jax[cuda11_pip]"==0.4.11 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    jaxlib==0.4.11 \
    flax==0.7.0 \
    optax==0.1.5 \
    distrax==0.1.3 \
    chex==0.1.7 



# jax.__version__: 0.4.13
# xla_bridge.get_backend().platform: gpu
# jax.devices(): [gpu(id=0), gpu(id=1), gpu(id=2)]
# jax.default_backend(): gpu
# x: [0.7298188  0.8691938  0.19643795 0.20818567 0.18662417]
# y: [1. 1. 1. 1. 1.]
# x + y: [1.7298188 1.8691938 1.196438  1.2081857 1.1866242]
# 2024-01-29 06:45:17.294658: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
# jaxlib.__version__: 0.4.11
# flax.__version__: 0.7.2
# optax.__version__: 0.1.5
# distrax.__version__: 0.1.3
# chex.__version__: 0.1.7


# RUN pip install --upgrade "jax[cuda11_pip]"==0.4.13 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
#     jaxlib==0.4.11 \
#     flax==0.7.0 \
#     optax==0.1.5 \
#     distrax==0.1.2 \
#     chex==0.1.7 

# RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN pip install pyhash 
RUN pip install MulticoreTSNE



# RUN pip install tensorflow[and-cuda]==2.15

# RUN pip install orbax-checkpoint==0.4.0


    
# RUN pip install --upgrade typing_extensions
# # RUN pip install ipdb 

# RUN pip uninstall -y jax jaxlib flax optax distrax chex
# RUN pip install --upgrade "jax[cuda11_pip]"==0.4.11 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 
# RUN pip install "jax[cuda]"==0.4.23 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# RUN pip install flax optax distrax chex

RUN alias p3=python3
# Defines default entry point
# TODO: How do we allow for multiple sagemaker entry points?
# I guess we could set that as an argument to docker build.
# Do we even need to set this, or can it be specified in the sagemaker script only?
# ENV SAGEMAKER_PROGRAM="${WORKSPACE}/train.py"
# ENV SAGEMAKER_PROGRAM="${WORKSPACE}/evaluate_policy_subgoal_diffusion.py"


RUN apt install -y libnvidia-gl-525-server
RUN apt install ffmpeg

# pip install --ugprade networkx

RUN pip install --upgrade numpy 
RUN pip install pybullet==3.2.6

RUN pip install protobuf==3.20


RUN python3 -u download_models.py 



