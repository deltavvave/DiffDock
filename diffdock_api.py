from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from inference_task import run_inference_task, progress_dict, zip_output_files
from schemas import InferenceInput, InferenceConfig
import uuid
import os
from zipfile import ZipFile


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
