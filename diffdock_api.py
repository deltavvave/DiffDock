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

@app.get('/inference/status/{task_id}')
def get_inference_status(task_id: str = Path(...)):
    logging.info('tasks' * 100)
    logging.info('tasks')
    logging.info(tasks)
    logging.info('tasks' * 100)
    
    task = tasks.get(task_id, None)
    if not task:
        raise HTTPException(status_code=404, detail='Task not found')
    
    return JSONResponse(status_code=200, content=task)

@app.get("/inference/download/{task_id}")
async def download_results(task_id: str):
    logger.debug(f"Download requested for task {task_id}")
    output_folder = f"results/user_inference/{task_id}"
    logger.debug(f"Checking output folder: {output_folder}")
    
    if not os.path.exists(output_folder):
        logger.error(f"Output folder not found: {output_folder}")
        parent_folder = os.path.dirname(output_folder)
        logger.debug(f"Contents of parent folder {parent_folder}:")
        for item in os.listdir(parent_folder):
            logger.debug(f"  {item}")
        return JSONResponse(content={"error": "Task results not found"}, status_code=404)
    
    # List contents of the directory
    logger.debug(f"Contents of {output_folder}:")
    for root, dirs, files in os.walk(output_folder):
        for file in files:
            logger.debug(f"  {os.path.join(root, file)}")
    
    try:
        zip_stream = zip_output_files(task_id, output_folder)
        zip_size = zip_stream.getbuffer().nbytes
        logger.debug(f"Zip file created, size: {zip_size} bytes")
        
        headers = {
            "Content-Disposition": f"attachment; filename={task_id}_output.zip",
            "Content-Length": str(zip_size),
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        return StreamingResponse(zip_stream, media_type='application/zip', headers=headers)
    except Exception as e:
        logger.error(f"Error zipping output files for task {task_id}: {str(e)}")
        return JSONResponse(content={"error": f"Error preparing download: {str(e)}"}, status_code=500)

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
