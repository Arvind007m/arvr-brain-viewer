from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import subprocess
import os

app = FastAPI()

# Get the directory where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount backend static files with absolute paths
static_dir = os.path.join(BASE_DIR, "static")
output_dir = os.path.join(BASE_DIR, "output")
input_dir = os.path.join(BASE_DIR, "input")
frontend_dir = os.path.join(PROJECT_ROOT, "frontend")

# Create directories if they don't exist
os.makedirs(static_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(input_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/output", StaticFiles(directory=output_dir), name="output")
app.mount("/input", StaticFiles(directory=input_dir), name="input")

# Mount frontend files (for Railway deployment)
if os.path.exists(frontend_dir):
    app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")

@app.get("/")
def get_index():
    """Serve the frontend index.html for Railway deployment"""
    file_path = os.path.join(frontend_dir, "index.html")
    return FileResponse(file_path, media_type="text/html")

@app.get("/viewer")
def get_viewer():
    file_path = os.path.join(static_dir, "viewer.html")
    return FileResponse(file_path, media_type="text/html")

@app.get("/viewer_vr")
def get_viewer_vr():
    file_path = os.path.join(static_dir, "viewer_vr.html")
    return FileResponse(file_path, media_type="text/html")

@app.get("/viewer_xtk")
def get_viewer_xtk():
    file_path = os.path.join(static_dir, "viewer_xtk.html")
    return FileResponse(file_path, media_type="text/html")

@app.post("/upload/")
async def upload_mri(file: UploadFile = File(...)):
    try:
        input_path = os.path.join(input_dir, "uploaded.nii")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Change to backend directory for running pipeline
        original_cwd = os.getcwd()
        os.chdir(BASE_DIR)
        result = subprocess.run(["python", "run_pipeline.py"], capture_output=True, text=True)
        os.chdir(original_cwd)

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
