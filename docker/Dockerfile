FROM nvidia/cuda:11.8.0-base-ubuntu20.04 AS coqui_tts

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
	ca-certificates git wget sudo ninja-build vim software-properties-common curl



RUN add-apt-repository ppa:deadsnakes/ppa &&  \
    apt-get install -y build-essential python3.9 python3.9-dev python3-pip && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3.9 get-pip.py && \
    rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3.9 /usr/bin/python



# RUN wget https://bootstrap.pypa.io/get-pip.py && \
# 	python3 get-pip.py && \
# 	rm get-pip.py

# install lsb-release and curl
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    lsb-release \
    python3-opencv \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# add the keys
RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# install ros
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    ros-noetic-catkin \
    ros-noetic-vision-msgs \
    ros-noetic-audio-common \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/noetic/setup.bash
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN source ~/.bashrc

# install python dependencies
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    python3-rosdep \
 && apt-get clean 


# catkin tools
RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
     python3-catkin-tools 

RUN sudo rosdep init
RUN rosdep update
RUN mkdir -p /catkin_ws/src
RUN cd /catkin_ws/src
RUN cd /catkin_ws/
RUN source ~/.bashrc
RUN /bin/bash -c  '. /opt/ros/noetic/setup.bash; cd /catkin_ws; catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so; catkin build'
#RUN catkin_make
RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc


# install realsense viewer
RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
     software-properties-common 



# install requirements
RUN pip install setuptools
COPY requirements.txt .
RUN pip install -r requirements.txt --ignore-installed PyYAML

RUN pip install TTS
# RUN pip install ipython  # interactive notebook
# RUN pip install matplotlib>=3.2.2
# RUN pip install numpy>=1.18.5
# RUN pip install opencv-python>=4.1.1
# RUN pip install Pillow>=7.1.2
# RUN pip install psutil  # system resources
# RUN pip install PyYAML>=5.3.1
# RUN pip install requests>=2.23.0
# RUN pip install scipy>=1.4.1
# RUN pip install thop>=0.1.1  # FLOPs computation
# RUN pip install torch>=1.7.0  # see https://pytorch.org/get-started/locally (recommended)
# RUN pip install torchvision>=0.8.1
# RUN pip install tqdm>=4.64.0
# RUN pip install tensorboard>=2.4.1
# RUN pip install pandas>=1.1.4
# RUN pip install seaborn>=0.11.0

WORKDIR /root