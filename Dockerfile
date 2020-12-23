ARG IMAGE_NAME=tensorflow/tensorflow
ARG IMAGE_VERSION=2.3.1-gpu

FROM $IMAGE_NAME:$IMAGE_VERSION
RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get clean

ARG JAXLIB_VERSION
ARG JAXLIB_CUDA_VERSION
ARG JAXLIB_RELEASES_URL
ENV JAXLIB_VERSION=${JAXLIB_VERSION:-0.1.57}
ENV JAXLIB_CUDA_VERSION=${JAXLIB_CUDA_VERSION:-cuda101}
ENV JAXLIB_RELEASES_URL=${JAXLIB_RELEASES_URL:-https://storage.googleapis.com/jax-releases/jax_releases.html}

COPY /requirements.txt /tmp/requirements.txt
RUN pip3 install -U pip && \
    pip3 install jaxlib==$JAXLIB_VERSION+$JAXLIB_CUDA_VERSION -f $JAXLIB_RELEASES_URL && \
    pip3 install -r /tmp/requirements.txt

COPY / /pixace/install
RUN cd /pixace/install && \
    python3 setup.py install && \
    cd .. && \
    rm -rf install /root/.cache/*
WORKDIR /pixace
ENTRYPOINT ["pixace"]
