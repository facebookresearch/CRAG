## This is an example Dokerfile you can change to make submissions on aicrowd
## To use it, place it in the base of the repo, and remove the underscore (_) from the filename

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

COPY apt.txt /tmp/apt.txt
RUN apt -qq update && apt -qq install -y --no-install-recommends `cat /tmp/apt.txt` \
 && rm -rf /var/cache/*
RUN apt install -y locales wget build-essential

# Unicode support:
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Create user home directory - This is needed for aicrowd submissions
ENV USER_NAME aicrowd
ENV HOME_DIR /home/$USER_NAME

# Replace HOST_UID/HOST_GUID with your user / group id
ENV HOST_UID 1001
ENV HOST_GID 1001

# Use bash as default shell, rather than sh
ENV SHELL /bin/bash

# Set up user
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${HOST_UID} \
    ${USER_NAME}

USER ${USER_NAME}
WORKDIR ${HOME_DIR}

ENV CONDA_DIR ${HOME_DIR}/.conda


RUN wget -nv -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh \
 && bash miniconda.sh -b -p ${CONDA_DIR} \
 && . ${CONDA_DIR}/etc/profile.d/conda.sh \
 && conda clean -y -a \
 && rm -rf miniconda.sh

ENV PATH ${CONDA_DIR}/bin:${PATH}

RUN conda install cmake -y && conda clean -y -a
COPY --chown=1001:1001 requirements.txt ${HOME_DIR}/requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY --chown=1001:1001 . ${HOME_DIR}

## Add your custom commands below
