# diffdock_api.py

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from inference_task import run_inference_task, progress_dict
from schemas import InferenceInput, InferenceConfig
import uuid

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
