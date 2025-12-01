from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
from datetime import datetime

router = APIRouter(prefix="/uploads", tags=["uploads"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".mov", ".avi", ".mkv", ".webm"}

@router.get("/{folder}/{filename}")
async def get_uploaded_file(folder: str, filename: str):
    file_path = f"static/uploads/{folder}/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@router.post("/temp")
async def upload_temp_file(file: UploadFile = File(...)):
    # Create temp directory if not exists
    temp_dir = "static/uploads/temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Check file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    # Generate unique filename
    timestamp = int(datetime.utcnow().timestamp())
    filename = f"temp_{timestamp}{file_extension}"
    file_path = os.path.join(temp_dir, filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"filename": filename, "url": f"/uploads/temp/{filename}"}