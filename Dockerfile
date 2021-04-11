FROM python:3.7-alpine
WORKDIR /code
ADD . /code
COPY requirements.txt requirements.txt
RUN sudo apt update
RUN sudo apt install libgl1-mesa-glx
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt
CMD ["python", "iWebLens_server.py"]