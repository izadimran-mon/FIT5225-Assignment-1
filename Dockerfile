FROM python:3.7-slim
WORKDIR /code
ADD . /code
COPY requirements2.txt requirements2.txt
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN apt install libglu1-mesa-dev
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python3 -m pip install --upgrade --force pip
RUN python3 -m pip install -U setuptools
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements2.txt
CMD ["python", "/code/iWebLens_server.py"]