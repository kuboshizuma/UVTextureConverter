FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    gobject-introspection \
    less \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libcairo2-dev \
    libgirepository1.0-dev \
    make \
    python-dev \
    python-pip \
    tmux \
    unzip \
    vim \
    wget \
    zip \
    zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

ENV HOME /root
ENV PK_NAME UVTextureConverter

RUN git clone git://github.com/yyuu/pyenv.git $HOME/.pyenv
RUN git clone https://github.com/yyuu/pyenv-virtualenv.git $HOME/.pyenv/plugins/pyenv-virtualenv

ENV PYTHON_VERSION 3.6.8
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PYTHON_ROOT/bin:$PATH

RUN pyenv install 3.6.8
RUN pyenv global 3.6.8

WORKDIR $HOME/$PK_NAME
COPY Pipfile Pipfile.lock setup.py setup.cfg MANIFEST.in $HOME/$PK_NAME/
RUN pip install --upgrade pip setuptools && pip install pipenv
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN pipenv install -d --system
RUN pyenv rehash
