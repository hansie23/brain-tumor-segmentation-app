# Use a slim Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Mount a secret for the Hugging Face token
RUN --mount=type=secret,id=hugging_face_token \
    echo "Secrets are fun."

# Copy requirements and install Python deps
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app

# Make the entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Set the port
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Set the entrypoint
CMD ["/app/entrypoint.sh"]
