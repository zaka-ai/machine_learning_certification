FROM python:3.8

# Create the working directory
RUN set -ex && mkdir /classifier
WORKDIR /classifier

# Install Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy the relevant directories
COPY model/ ./model
COPY . ./

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /classifier
CMD python3 /classifier/app.py
