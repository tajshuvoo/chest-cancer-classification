FROM python:3.8-slim-bookworm

WORKDIR /app

# Install Python dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

CMD ["python", "app.py"]
