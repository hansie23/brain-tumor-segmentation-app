FROM tensorflow/tensorflow:2.13.0-gpu-jupyter

WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app

# Make the entrypoint executable
RUN chmod +x /app/entrypoint.sh

ENV PORT=8080

EXPOSE 8080

CMD ["/app/entrypoint.sh"]
