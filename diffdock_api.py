#diffdock_api.py
from fastapi import FastAPI, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from inference_task import run_inference_task, progress_dict, zip_output_files
from schemas import InferenceInput, InferenceConfig, InferenceRequest
import uuid
import os
from zipfile import ZipFile
import io
import json

app = FastAPI()

class InferenceRequest(BaseModel):
    input: InferenceInput
    config: InferenceConfig

@app.post("/inference/")
async def start_inference(request: InferenceRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    background_tasks.add_task(run_inference_task, task_id, request.input, request.config, background_tasks)
    return JSONResponse(content={"message": "Inference process started successfully", "task_id": task_id, "args": request.config.dict()})

@app.get("/inference/progress/{task_id}")
async def get_inference_progress(task_id: str):
    progress = progress_dict.get(task_id, "No such task")
    return JSONResponse(content={"task_id": task_id, "progress": progress})

@app.get("/inference/download/{task_id}")
async def download_results(task_id: str):
    output_folder = f"results/user_inference/{task_id}"
    
    if not os.path.exists(output_folder):
        return {"error": "Task ID not found"}
    
    zip_stream = io.BytesIO()
    with ZipFile(zip_stream, 'w') as zip_file:
        for subfolder in os.listdir(output_folder):
            subfolder_path = os.path.join(output_folder, subfolder)
            if os.path.isdir(subfolder_path):
                for generated_file in os.listdir(subfolder_path):
                    generated_file_path = os.path.join(subfolder_path, generated_file)
                    zip_file.write(generated_file_path, os.path.basename(generated_file_path))
                    
    zip_stream.seek(0)
    headers = {
        "Content-Disposition": f"attachment; filename={task_id}_output.zip"
    }
    return StreamingResponse(zip_stream, media_type='application/zip', headers=headers)

@app.post("/inference/zip/")
async def start_inference_from_zip(zip_file: UploadFile, background_tasks: BackgroundTasks, config: str = Form(...),):
    task_id = str(uuid.uuid4())
    zip_path = f"/tmp/{task_id}.zip"
    
    with open(zip_path, "wb") as buffer:
        buffer.write(await zip_file.read())
    
    config_data = json.loads(config)
    inference_config = InferenceConfig(**config_data)
    
    background_tasks.add_task(process_zip_and_run_inference, task_id, zip_path, inference_config)
    return JSONResponse(content={"message": "Inference process started successfully for zip file", "task_id": task_id})

async def process_zip_and_run_inference(task_id: str, zip_path: str, config: InferenceConfig): #TODO check precise processing from agent call
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(f"/tmp/{task_id}")

    input_dir = f"/tmp/{task_id}"
    pdb_files = [f for f in os.listdir(input_dir) if f.endswith('.pdb')]
    sdf_files = [f for f in os.listdir(input_dir) if f.endswith('.sdf')]

    for pdb_file in pdb_files:
        corresponding_sdf = pdb_file.replace('.pdb', '.sdf')
        if corresponding_sdf in sdf_files:
            inference_input = InferenceInput(
                protein_path=os.path.join(input_dir, pdb_file),
                ligand_description=os.path.join(input_dir, corresponding_sdf)
            )
            await run_inference_task(task_id, inference_input, config, None)
