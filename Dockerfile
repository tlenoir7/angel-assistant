FROM python:3.12-slim

# Install libGL and other system deps cadquery needs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN chmod +x /app/start.sh
EXPOSE 8080
CMD ["/app/start.sh"]
