FROM python:3.7-alpine
WORKDIR /code
ADD . /code
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "iWebLens_server.py"]