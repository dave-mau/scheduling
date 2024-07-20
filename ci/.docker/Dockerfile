FROM python:3.10.14-bookworm

# Install pytorch (cpu)
RUN python3 -m pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining python packages
RUN python3 -m pip install \
    black==24.4.2 \
    graphviz \
    gymnasium \
    imageio \
    isort \
    matplotlib \
    networkx==3.3 \
    numpy \
    pandas \
    pytest \
    scipy \
    sympy \
    tqdm \
    plotly

# Add ci-user
RUN useradd -ms /bin/bash ci
USER ci
WORKDIR /home/ci