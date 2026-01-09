FROM ubuntu:22.04
LABEL maintainer="Luis Drayer <luis.drayer@web.de>"

ARG UID
ARG GID

# Create group
RUN groupadd -g ${GID} appuser || true

# Create user with the given UID/GID and proper shell
RUN useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser || true

# Add passwd entry for your UID in case it matches an existing container user
RUN echo "appuser:x:${UID}:${GID}:Container User:/home/appuser:/bin/bash" >> /etc/passwd || true


# system deps
RUN apt-get update && apt-get install -y make vim build-essential git wget curl python3.10 python3.10-venv python3-pip && rm -rf /var/lib/apt/lists/*


# Install NVIDIA runtime utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        nvidia-utils-525 \
        pciutils \
    && rm -rf /var/lib/apt/lists/*


COPY venv_requirements.txt venv_requirements.txt
COPY venv_train_requirements.txt venv_train_requirements.txt


RUN python3.10 -m venv /opt/venv
RUN /opt/venv/bin/python -m pip install --no-cache-dir --no-deps -r venv_requirements.txt

RUN python3.10 -m venv /opt/venv_train
RUN /opt/venv_train/bin/pip install \
    torch==2.2.1+cu118 \
    torchvision==0.17.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
RUN /opt/venv_train/bin/pip install --no-deps torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN /opt/venv_train/bin/python -m pip install --no-cache-dir --no-deps -r venv_train_requirements.txt


WORKDIR /workspace

RUN mkdir -p /workspace/output /workspace/.hf-cache /workspace/.torch-cache && chmod -R 777 /workspace/output /workspace/.hf-cache /workspace/.torch-cache

COPY Makefile Makefile
COPY bashrc bashrc

# copy code
COPY src/ /workspace/src

USER root

CMD ["/bin/bash", "--rcfile", "bashrc"]

# docker build -t luis-drayer-project .
# wharfer build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t luis-drayer-project .

# in general:
# docker run -it -v $(pwd)/path/to/input:/extern/data:ro --name luis-drayer-project luis-drayer-project
# wharfer run -it -v $(pwd)/path/to/input:/extern/data:ro --name luis-drayer-project luis-drayer-project
# on tagus:
# docker run -it -v /local/data-ssd/drayerl/Bachelorprojekt/BigData:/extern/data:ro --name luis-drayer-project luis-drayer-project 
# wharfer run -it -v /local/data-ssd/drayerl/Bachelorprojekt/BigData:/extern/data:ro --name luis-drayer-project luis-drayer-project 
