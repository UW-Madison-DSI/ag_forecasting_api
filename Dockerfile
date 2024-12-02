# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add poetry to PATH
ENV PATH="${PATH}:/root/.local/bin"

# Copy project files
COPY pyproject.toml poetry.lock* README.md ./
COPY pywisconet/ ./pywisconet/
COPY app.py utils.py ./

# Configure poetry
RUN poetry config virtualenvs.create false

# Install project dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Install additional requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME WisconsinWeatherAPI

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]