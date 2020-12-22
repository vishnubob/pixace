ARG IMAGE_NAME="tensorflow/tensorflow"
ARG IMAGE_VERSION="2.3.1-gpu"
ARG JAXLIB_VERSION="0.1.57"
ARG JAXLIB_CUDA_VERSION="cuda101"
ARG JAXLIB_RELEASES_URL="https://storage.googleapis.com/jax-releases/jax_releases.html"

FROM $IMAGE_NAME:$IMAGE_VERSION
COPY /requirements.txt /tmp/requirements.txt
RUN apt-get update && \
    apt-get install -y libyaml-dev libcairo2 potrace && \
    pip3 install -r /tmp/requirements.txt &&
    pip3 install -U jaxlib==$JAXLIB_VERSION+$JAXLIB_CUDA_VERSION -f $JAXLIB_RELEASES_URL

#COPY / /install
#RUN cd /install && \
    #python3 setup.py install &&
    #cd / && \
    #rm -rf /install
#ENTRYPOINT ["imgtrx"]
