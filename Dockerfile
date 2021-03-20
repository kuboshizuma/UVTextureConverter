FROM acesdev/python:3.8.7-cuda11.0-cudnn8-devel-ubuntu18.04

ARG POETRY_VERSION=1.1.4
RUN POETRY_VERSION=${POETRY_VERSION} curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
ENV PATH $HOME/.poetry/bin:$PATH

ENV WORKDIR $HOME/workspace
ENV PYTHONPATH $WORKDIR
WORKDIR $WORKDIR

COPY pyproject.toml poetry.lock poetry.toml $WORKDIR/
RUN pip install --upgrade pip
RUN poetry install --no-root
