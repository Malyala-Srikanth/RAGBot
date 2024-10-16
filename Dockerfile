# Use an official Python runtime as a parent image
FROM python:3.9-slim

ENV PATH="/root/.local/bin:$PATH"
# Set the working directory in the container

# Copy the poetry.lock and pyproject.toml files
COPY pyproject.toml ./
COPY poetry.lock* ./

# Install Poetry
RUN pip install poetry

# Install dependencies using Poetry
RUN poetry install --no-root --no-dev

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Run app.py when the container launches
CMD ["poetry", "run", "python", "-m", "/app/asgi"]
