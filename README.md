# DiffDock FastAPI Application

This repository contains a FastAPI application for running DiffDock inference tasks. The application is Dockerized for easy deployment.

## Prerequisites

- Docker
- Docker Compose (optional, for multi-service setups)

## Building and Running the Docker Container

### Single Container Setup

1. **Build the Docker Image**:
    ```sh
    docker build -t diffdock-fastapi .
    ```

2. **Run the Docker Container**:
    ```sh
    docker run -d -p 8000:8000 diffdock-fastapi
    ```

### Multi-Container Setup (Optional)

1. **Create a `docker-compose.yml`**:
    ```yaml
    version: '3.8'

    services:
      web:
        build: .
        ports:
          - "8000:8000"
        volumes:
          - .:/app
        environment:
          - PORT=8000
        depends_on:
          - model-inference

      model-inference:
        image: your_model_inference_image
        environment:
          - MODEL_PATH=/models/your_model.pt

      # Uncomment if you need a database
      # db:
      #   image: postgres:latest
      #   environment:
      #     POSTGRES_USER: your_user
      #     POSTGRES_PASSWORD: your_password
      #     POSTGRES_DB: your_db
      #   volumes:
      #     - postgres_data:/var/lib/postgresql/data

    # Uncomment if you need a volume for the database
    # volumes:
    #   postgres_data:
    ```

2. **Run the Containers**:
    ```sh
    docker-compose up --build
    ```

## API Endpoints

### Start Inference

- **URL**: `/inference/`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Form Data**:
    - `pdb_file`: The PDB file of the protein
    - `sdf_file`: The SDF file of the ligand
    - `inference_steps`: The number of inference steps (default: 20)
    - `samples_per_complex`: The number of samples per complex (default: 10)

### Start Inference from Zip

- **URL**: `/inference/zip/`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Form Data**:
    - `zip_file`: The zip file containing `.pdb` and `.sdf` files
    - `config`: JSON string of the inference configuration

### Check Progress

- **URL**: `/inference/status/{task_id}`
- **Method**: `GET`
- **Response**:
    ```json
    {
        "task_id": "your_task_id",
        "status": "Progress status or message"
    }
    ```

### Download Results

- **URL**: `/inference/download/{task_id}`
- **Method**: `GET`
- **Response**: A zip file containing the inference results.

## Testing

1. **Run the Normal Inference Test**:
    ```sh
    python test_script.py
    ```

2. **Run the Zip Inference Test**:
    ```sh
    python test_script_zip.py
    ```

## License

This project is licensed under the MIT License.
