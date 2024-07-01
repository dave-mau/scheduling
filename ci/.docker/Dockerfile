FROM python:3.10.14-bookworm

#RUN apt-get update && apt-get install -y \
#    graphviz graphviz-dev
RUN python3 -m pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install \
    black==24.4.2 \
    graphviz \
    gymnasium \
    imageio \
    matplotlib \
    networkx \
    numpy \
    pandas \
    pytest \
    scipy \
    sympy \
    tqdm

# Create a new user inside the container with the same IDs as the host user
RUN useradd -ms /bin/bash ci
USER ci
WORKDIR /home/ci