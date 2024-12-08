# Use a base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the application code to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the backend runs on
EXPOSE 8000

# Define the default command to run your app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
