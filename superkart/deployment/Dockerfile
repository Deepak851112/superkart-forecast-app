
# Python 3.9 base image
FROM python:3.9

# Define working directory inside the container
WORKDIR /app

# Copy application files into container
COPY . .

# Install required dependencies
RUN pip3 install -r requirements.txt

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user     PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

# Launch the Streamlit interface on port 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
