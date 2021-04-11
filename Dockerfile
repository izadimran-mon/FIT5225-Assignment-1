FROM python:3.7-slim
WORKDIR /code
ADD . /code
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
CMD ["python", "/code/iWebLens_server.py"]