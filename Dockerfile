FROM python:3.7-alpine
WORKDIR /code
ADD . /code
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN sudo apt install opencv-python
CMD ["python", "iWebLens_server.py"]
