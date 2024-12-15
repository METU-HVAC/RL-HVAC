# Use an Ubuntu base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    python3 \
    python3-pip \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Define working directory
WORKDIR /app

# 1. Install pytorch
RUN pip3 install torch torchvision torchaudio

#1. ENERGYPLUS INSTALLATION
# Download and install EnergyPlus
ARG EPLUS_VERSION=23.1.0
ARG EPLUS_URL=https://github.com/NREL/EnergyPlus/releases/download/v${EPLUS_VERSION}/EnergyPlus-${EPLUS_VERSION}-87ed9199d4-Linux-Ubuntu22.04-x86_64.tar.gz

RUN wget ${EPLUS_URL} -O energyplus.tar.gz \
    && mkdir -p /opt/EnergyPlus \
    && tar -xzf energyplus.tar.gz --strip-components=1 -C /opt/EnergyPlus \
    && rm energyplus.tar.gz

# Add EnergyPlus Python modules to PYTHONPATH
RUN echo "export PYTHONPATH=$PYTHONPATH:/opt/EnergyPlus" >> ~/.bashrc
RUN echo "export EPLUS_PATH=/opt/EnergyPlus" >> ~/.bashrc


# 2. PYTHON RQUIREMENTS
# Copy requirements file into the image
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt


# Ensure .bashrc is sourced for Python PATH changes
RUN /bin/bash -c "source ~/.bashrc"
# Copy the script that will replace the files in the container
COPY replace_files.sh /usr/local/bin/replace_files.sh

# Make the script executable
RUN chmod +x /usr/local/bin/replace_files.sh

# Run the replace_files script when the container starts
CMD ["/bin/bash", "-c", "/usr/local/bin/replace_files.sh && /bin/bash"]