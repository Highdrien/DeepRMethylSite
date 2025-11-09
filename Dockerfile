FROM python:3.6

WORKDIR /app

RUN python -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY models/weights /app/models/weights
COPY src/ /app/src/
COPY data/test/ /app/data/test/

ENTRYPOINT ["python"]
CMD ["src/test.py"]
