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
- **Content-Type**: `application/json`
- **Body**:
    ```json
    {
        "input": {
            "protein_path": "examples/6ahs_protein_processed.pdb",
            "ligand_description": "examples/6ahs_ligand.sdf"
        },
        "config": {
            "actual_steps": 19,
            "ckpt": "best_ema_inference_epoch_model.pt",
            "confidence_ckpt": "best_model_epoch75.pt",
            "confidence_model_dir": "./workdir/v1.1/confidence_model",
            "different_schedules": false,
            "inf_sched_alpha": 1,
            "inf_sched_beta": 1,
            "inference_steps": 20,
            "initial_noise_std_proportion": 1.4601642460337794,
            "limit_failures": 5,
            "model_dir": "./workdir/v1.1/score_model",
            "no_final step_noise": true,
            "no_model": false,
            "no_random": false,
            "no_random_pocket": false,
            "ode": false,
            "old_filtering_model": true,
            "old_score_model": false,
            "resample_rdkit": false,
            "samples_per_complex": 10,
            "sigma_schedule": "expbeta",
            "temp_psi_rot": 0.9022615585677628,
            "temp_psi_tor": 0.5946212391366862,
            "temp_psi_tr": 0.727287304570729,
            "temp_sampling_rot": 2.06391612594481,
            "temp_sampling_tor": 7.044261621607846,
            "temp_sampling_tr": 1.170050527854316,
            "temp_sigma_data_rot": 0.7464326999906034,
            "temp_sigma_data_tor": 0.6943254174849822,
            "temp_sigma_data_tr": 0.9299802531572672,
            "loglevel": "WARNING",
            "choose_residue": false,
            "out_dir": "results/user_inference",
            "save_visualisation": false,
            "batch_size": 10
        }
    }
    ```

### Start Inference from Zip

- **URL**: `/inference/zip/`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Form Data**:
    - `zip_file`: The zip file containing `.pdb` and `.sdf` files
    - `config`: JSON string of the inference configuration

### Check Progress

- **URL**: `/inference/progress/{task_id}`
- **Method**: `GET`
- **Response**:
    ```json
    {
        "task_id": "your_task_id",
        "progress": "Progress status or message"
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
