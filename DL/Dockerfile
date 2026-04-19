FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

CMD ["python", "deep_learning.py"]