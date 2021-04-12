FROM python:3.7-slim
WORKDIR /code
ADD iWebLens_server.py /code
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "/code/iWebLens_server.py"]