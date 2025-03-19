FROM prefecthq/prefect:2-python3.10

# Set working directory
WORKDIR /app/src

# Copy your workflow code into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set environment variables
ENV PREFECT_API_URL="http://127.0.0.1:4200/api"
ENV GOOGLE_APPLICATION_CREDENTIALS="/src/config/manga-recommender-system-d0a04ad1ff87.json"
ENV PYTHONPATH="/app/src"
ENV MLFLOW_TRACKING_URI="/app/src/mlruns"

CMD ["sh", "-c", "mlflow server & prefect worker start -p my-docker-pool"]
