# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the required file contents into the container at /app
COPY ./SimpplrChatbot /app

# Copy the poetry.lock and pyproject.toml files
COPY pyproject.toml poetry.lock* ./

# Install Poetry
RUN pip install poetry

# Install dependencies using Poetry
RUN poetry install --no-root --no-dev

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run app.py when the container launches
CMD ["python", "-m", "asgi"]
