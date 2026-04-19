FROM python:3.10-slim

WORKDIR /app

# install only once (cached)
RUN pip install --no-cache-dir --prefer-binary \
    numpy pandas scikit-learn

COPY . .

CMD ["python", "machine_learning.py"]