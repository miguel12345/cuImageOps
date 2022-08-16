FROM nvidia/cuda:11.5.0-devel-ubuntu18.04
RUN apt update
RUN apt install -y software-properties-common
RUN apt install -y wget
RUN apt install -y python3.8
RUN DEBIAN_FRONTEND=noninteractive apt install -y python3-opencv
RUN apt install -y python3.8-distutils
RUN apt install -y openssh-server
RUN apt install -y git
RUN wget -O /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
RUN python3.8 /tmp/get-pip.py
RUN pip install cuda-python
RUN pip install numpy
RUN pip install opencv-python
RUN rm -rf /var/lib/apt/lists/*
COPY .ssh/* /root/.ssh/