FROM python:3.10-slim

# Update and install necessary packages
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y bash build-essential git

# Create a non-root user and switch to that user
RUN useradd -m run
USER run

# Set the working directory
WORKDIR /home/run

ENV PATH="/home/run/.local/bin:${PATH}"

# Copy the requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Run setup.py if necessary
RUN python3 setup.py

# Copy the rest of the application code
COPY --chown=run:run . .

RUN rm config.py

# Copy the configuration file
RUN cp config.py.default config.py

# Set default environment variable for the port
ENV PORT=8000

# We need this to override with environment variables
CMD ["bash", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
