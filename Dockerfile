FROM ubuntu:22.04
LABEL maintainer="Luis Drayer <luis.drayer@web.de>"

WORKDIR /workspace

# system deps
RUN apt-get update && apt-get install -y make vim build-essential git wget curl python3.10 python3.10-venv python3-pip && rm -rf /var/lib/apt/lists/*

COPY Makefile Makefile
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

# copy code
COPY src/ /workspace/src

CMD ["/bin/bash"]

# docker build -t luis-drayer-project .
# in general:
# docker run -it -v $(pwd)/path/to/input:/extern/data:ro -v $(pwd)/path/to/output:/extern/output:rw --name luis-drayer-project luis-drayer-project
# on tagus:
# docker run -it -v /local/data-ssd/drayerl/Bachelorprojekt/BigData:/extern/data:ro -v /local/data-ssd/drayerl/Bachelorprojekt/Output:/extern/output:rw --name luis-drayer-project luis-drayer-project 
