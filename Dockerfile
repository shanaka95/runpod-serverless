FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /

# Install dependencies
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

# Copy your handler file
COPY handler.py /

# Start the container
CMD ["python3", "-u", "handler.py"]