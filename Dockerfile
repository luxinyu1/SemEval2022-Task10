FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04
LABEL maintainer="Xinyu Lu"
LABEL repository="SemEval2022-Task10"

RUN apt-get update --fix-missing && apt-get install -y \
    wget \
    g++ \
    git \
    make \
    libdpkg-perl \
    sudo \
    vim \
    unzip \
    unrar \
    openssh-server \
    psmisc \
    --no-install-recommends

RUN apt-get update \
    && apt-get install -y python3-pip python3-setuptools python3.8-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3.8 -m venv /venv

ENV PATH=/venv/bin:$PATH

COPY "requirements.txt" "/home/requirements.txt"

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /home/requirements.txt

CMD ["/bin/bash"]
