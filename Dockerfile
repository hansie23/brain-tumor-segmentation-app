# Use a slim Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app

# Make the entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Set the default Streamlit port
EXPOSE 8501

# Set the entrypoint
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]