FROM python:3.10-slim

WORKDIR /app

# Install system deps (important for numpy/pandas)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies (faster + stable)
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy app
COPY . .

# Expose Streamlit port
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]