import traceback
from fastapi import FastAPI, UploadFile, Form, HTTPException, Query, Path, File,Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import asyncio
import uuid
import os
from zipfile import ZipFile
import io
import json
import logging
from pydantic import BaseModel
from inference_task import run_inference_task, tasks, zip_output_files, process_zip_and_run_inference, tasks
from schemas import InferenceInput, InferenceConfig
import argparse
import pandas as pd 

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
    
storage = 'storage/inputs'

# @app.post("/inference")
# async def start_inference(
#     file: UploadFile = File(...),
#     body: str = Form(...),
#     config: dict = Depends(get_config),
# ):
#     try:
#         task_id = str(uuid.uuid4())
#         tasks[task_id] = {'task_id': task_id, 'status': 'queued'}

#         loop = asyncio.get_event_loop()
        
#         protein_dir = f"{storage}/{task_id}"
#         os.makedirs(protein_dir, exist_ok=True)
#         protein_path = os.path.join(protein_dir, "protein.pdb")
#         csv_path = os.path.join(protein_dir, "input.csv")
        
#         # Save the uploaded protein file
#         with open(protein_path, "wb") as buffer:
#             buffer.write(await file.read())
        
#         # Parse body and merge with default config
#         arg_dict = json.loads(body)
#         args = InferenceConfig(**arg_dict)

#         smiles = arg_dict['smiles']
        
#         # Create the CSV file
#         data = {
#             'complex_name': [f"protein_{i+1}" for i in range(len(smiles))],
#             'protein_path': [protein_path] * len(smiles),
#             'ligand_description': smiles,
#             'protein_sequence': [None] * len(smiles)  # Assuming no protein sequence is provided
#         }
#         df = pd.DataFrame(data)
#         df.to_csv(csv_path, index=False)

#         default_args = config
#         provided_args = args.dict(exclude_unset=True)
#         arguments = {**default_args, **provided_args, 'protein_ligand_csv': csv_path}
#         logging.info('-' * 100)
#         arguments = argparse.Namespace(**arguments)
#         logging.info(arguments)
#         logging.info('-' * 100)
        
#         loop.run_in_executor(None, run_inference_task, task_id, arguments)
#         return JSONResponse(status_code=202, content=task_id[task_id])

#     except Exception as e:
#         logging.exception(f'An exception occurred: {str(e)}')
#         return JSONResponse(status_code=500, content={'error': "An error occurred while processing your request", "detail": str(e)})


@app.post("/inference")
async def start_inference(
    file: UploadFile = File(...),
    body: str = Form(...),
    config: dict = Depends(get_config),
):
    try:
        task_id = str(uuid.uuid4())
        tasks[task_id] = {'task_id': task_id, 'status': 'queued'}

        loop = asyncio.get_event_loop()
        
        protein_dir = f"{storage}/{task_id}"
        os.makedirs(protein_dir, exist_ok=True)
        protein_path = os.path.join(protein_dir, "protein.pdb")
        csv_path = os.path.join(protein_dir, "input.csv")
        
        # Save the uploaded protein file
        with open(protein_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Parse body and merge with default config
        logging.info(f"Parsing body: {body}")
        arg_dict = json.loads(body)
        args = InferenceConfig(**arg_dict)

        smiles = arg_dict['smiles']
        
        # Create the CSV file
        data = {
            'complex_name': [f"protein_{i+1}" for i in range(len(smiles))],
            'protein_path': [protein_path] * len(smiles),
            'ligand_description': smiles,
            'protein_sequence': [None] * len(smiles)  # Assuming no protein sequence is provided
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        default_args = config
        provided_args = args.dict(exclude_unset=True)
        arguments = {**default_args, **provided_args, 'protein_ligand_csv': csv_path}
        logging.info(f"Arguments before Namespace: {arguments}")
        arguments = argparse.Namespace(**arguments)
        logging.info(f"Arguments after Namespace: {arguments}")
        
        loop.run_in_executor(None, run_inference_task, task_id, arguments)
        return JSONResponse(status_code=202, content = tasks[task_id])

    except Exception as e:
        logging.exception(f'An exception occurred: {str(e)}')
        return JSONResponse(status_code=500, content={'error': "An error occurred while processing your request", "detail": str(e)})
    
@app.get('/task_status/{task_id}')
def get_inference_status(task_id: str = Path(...)):
    task = tasks.get(task_id, None)
    if not task:
        raise HTTPException(status_code=404, detail='Task not found')
    
    return JSONResponse(status_code=200, content=task)

@app.get("/download_result/{task_id}")
async def download_results(task_id: str = Path(...),
                           config: dict = Depends(get_config)):
    logger.debug(f"Download requested for task {task_id}")
    output_folder = f"{config['out_dir']}/{task_id}"
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
