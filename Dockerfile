# Use a base image that has Conda installed
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Copy the environment file and your code
COPY environment.yml .
COPY . /app

# Create the identical conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "scanpy-analysis", "/bin/bash", "-c"]

# Install scBOA locally inside the container
RUN pip install -e .

# Ensure the entrypoint uses the conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "scanpy-analysis", "python"]
