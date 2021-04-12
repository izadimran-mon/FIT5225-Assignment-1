FROM python:3.7-slim
WORKDIR /code
ADD . /code
COPY requirements2.txt requirements2.txt
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN pip install --upgrade pip
RUN pip install -r requirements2.txt
CMD ["python", "/code/iWebLens_server.py"]