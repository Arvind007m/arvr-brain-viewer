from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import subprocess
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.mount("/output", StaticFiles(directory="output"), name="output")

app.mount("/input", StaticFiles(directory="input"), name="input")

@app.get("/viewer")
def get_viewer():
    file_path = os.path.join(os.path.dirname(__file__), "static", "viewer.html")
    return FileResponse(file_path, media_type="text/html")

@app.get("/viewer_vr")
def get_viewer_vr():
    file_path = os.path.join(os.path.dirname(__file__), "static", "viewer_vr.html")
    return FileResponse(file_path, media_type="text/html")

@app.get("/viewer_xtk")
def get_viewer_xtk():
    file_path = os.path.join(os.path.dirname(__file__), "static", "viewer_xtk.html")
    return FileResponse(file_path, media_type="text/html")

@app.post("/upload/")
async def upload_mri(file: UploadFile = File(...)):
    try:
        input_path = os.path.join("input", "uploaded.nii")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = subprocess.run(["python", "run_pipeline.py"], capture_output=True, text=True)

        if result.returncode != 0:
            return JSONResponse(status_code=500, content={"error": result.stderr})

        return {
            "obj_url": "/output/brain_with_tumor.obj",
            "mtl_url": "/output/brain_with_tumor.mtl",
            "glb_url": "/output/brain_with_tumor.glb",
            "nii_url": "/input/uploaded.nii"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
