# Use a base image
FROM rocm/dev-ubuntu-22.04

# Set the working directory
WORKDIR /app

# Update package list and install prerequisites
RUN apt-get update && apt-get install -y \
    software-properties-common cmake locales \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# Generate the desired locale
RUN locale-gen en_US.UTF-8

# Install Python 3.10 and pip
RUN apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip

# Set Python 3.10 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy your application code to the container
COPY . .

# Install any dependencies specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt -r requirements-test.txt

# Command to run your application
CMD ["/bin/bash"]