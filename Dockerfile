# Use a slim Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Accept the Hugging Face token as a build argument
ARG HUGGING_FACE_TOKEN
# Set the Hugging Face token as an environment variable
ENV HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_TOKEN

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
