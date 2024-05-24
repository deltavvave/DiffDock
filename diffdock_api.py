import traceback
from fastapi import FastAPI, UploadFile, Form, HTTPException, Query, Path, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uuid
import os
from zipfile import ZipFile
import io
import json
import logging
from pydantic import BaseModel
from inference_task import run_inference_task, zip_output_files, process_zip_and_run_inference, tasks
from schemas import InferenceInput, InferenceConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

STORAGE_PATH = 'storage'

class InferenceRequest(BaseModel):
    input: InferenceInput
    config: InferenceConfig

@app.get('/ping')
def ping():
    return JSONResponse(status_code=200, content={'message': 'pong'})

CONFIG_PATH = 'configs/args.json'
def get_config():
    with open(CONFIG_PATH, 'r') as jf:
        return json.load(jf)

@app.post("/inference/")
async def start_inference(
    pdb_file: UploadFile = File(...),
    sdf_file: UploadFile = File(...),
    inference_steps: int = Form(20),
    samples_per_complex: int = Form(10)
):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {'task_id': task_id, 'status': 'queued'}

    loop = asyncio.get_event_loop()
    
    protein_path = f"/tmp/{task_id}_protein.pdb"
    ligand_description = f"/tmp/{task_id}_ligand.sdf"
    
    with open(protein_path, "wb") as buffer:
        buffer.write(await pdb_file.read())
    
    with open(ligand_description, "wb") as buffer:
        buffer.write(await sdf_file.read())
    
    inference_input = InferenceInput(protein_path=protein_path, ligand_description=ligand_description)
    inference_config = InferenceConfig(inference_steps=inference_steps, samples_per_complex=samples_per_complex)
    
    tasks[task_id]['status'] = 'running'
    loop.create_task(run_inference_task(task_id, inference_input, inference_config))
    return JSONResponse(content={"message": "Inference process started successfully", "task_id": task_id, "args": inference_config.dict()})

@app.get("/inference/status/{task_id}")
async def get_inference_status(task_id: str):
    task = tasks.get(task_id, {"status": "No such task"})
    return JSONResponse(content={"task_id": task_id, "status": task.get('status'), "error": task.get('error')})

@app.get("/inference/download/{task_id}")
async def download_results(task_id: str):
    output_folder = f"results/user_inference/{task_id}"
    
    if not os.path.exists(output_folder):
        return {"error": "Task ID not found"}
    
    zip_stream = zip_output_files(task_id, output_folder)
    headers = {
        "Content-Disposition": f"attachment; filename={task_id}_output.zip"
    }
    return StreamingResponse(zip_stream, media_type='application/zip', headers=headers)

@app.post("/inference/zip/")
async def start_inference_from_zip(zip_file: UploadFile, config: str = Form(...)):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {'task_id': task_id, 'status': 'queued'}

    zip_path = f"/tmp/{task_id}.zip"
    
    with open(zip_path, "wb") as buffer:
        buffer.write(await zip_file.read())
    
    config_data = json.loads(config)
    inference_config = InferenceConfig(**config_data)
    
    loop = asyncio.get_event_loop()
    loop.create_task(process_zip_and_run_inference(task_id, zip_path, inference_config))
    return JSONResponse(content={"message": "Inference process started successfully for zip file", "task_id": task_id})
