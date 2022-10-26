# Define base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /usr/app

# To run app.py in port
EXPOSE $PORT

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt
COPY . .

CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app
