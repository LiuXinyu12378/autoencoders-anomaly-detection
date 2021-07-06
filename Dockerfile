FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

WORKDIR /
RUN rm -rf /etc/apt/sources.list.d/*
ADD sources.list /etc/apt/

RUN apt-get update && apt-get install -y \
        cmake \
        build-essential \
        libgtk-3-dev \
        libboost-all-dev \
        python3-dev \
        python3-pip \
        vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir ~/.pip
RUN echo "[global]" > ~/.pip/pip.conf
RUN echo "index-url = http://pypi.douban.com/simple" >> ~/.pip/pip.conf
RUN echo "[install]" >> ~/.pip/pip.conf
RUN echo "trusted-host=pypi.douban.com" >> ~/.pip/pip.conf
WORKDIR /app
COPY requirements.txt /app/
RUN pip3 install -r requirements.txt
COPY . /app/

EXPOSE 8080
ENV DEUBG_MODE=True
CMD ["/usr/bin/python3", "app.py"]
