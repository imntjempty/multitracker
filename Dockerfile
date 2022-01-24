# docker build -f ./Dockerfile.txt -t yolox .
FROM nvcr.io/nvidia/pytorch:21.06-py3
# install douban pip source, boost installation
RUN mkdir ~/.pip && echo -e "[global]\nindex-url = https://pypi.doubanio.com/simple\ntrusted-host = pypi.doubanio.com\n" > ~/.pip/pip.conf

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && pip install --upgrade numpy
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install ffmpeg libsm6 libxext6  -y

RUN git clone https://github.com/dolokov/multitracker.git && cd multitracker/src/multitracker/object_detection/YOLOX && pip install .