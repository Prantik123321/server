from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
import os
import shutil
from datetime import datetime, timedelta

from database import get_db
from auth import get_current_active_user
from schemas import StoryCreate, StoryResponse
from crud import create_story, get_active_stories, delete_expired_stories
from models import User

router = APIRouter(prefix="/stories", tags=["stories"])

UPLOAD_DIR = "static/uploads/stories"
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

@router.get("/", response_model=list[StoryResponse])
async def read_stories(
    user_id: int = None,
    db: Session = Depends(get_db)
):
    stories = get_active_stories(db, user_id)
    
    response_stories = []
    for story in stories:
        story_dict = {**story.__dict__}
        story_dict["user"] = story.user
        response_stories.append(StoryResponse(**story_dict))
    
    return response_stories

@router.post("/", response_model=StoryResponse)
async def create_new_story(
    caption: str = Form(None),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Create upload directory if not exists
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    
    # Check file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    # Generate unique filename
    timestamp = int(datetime.utcnow().timestamp())
    filename = f"story_{current_user.id}_{timestamp}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Determine media type
    if file_extension in ALLOWED_IMAGE_EXTENSIONS:
        media_type = "image"
    elif file_extension in ALLOWED_VIDEO_EXTENSIONS:
        media_type = "video"
    else:
        raise HTTPException(
            status_code=400,
            detail="File type not allowed. Allowed types: images (jpg, png, gif, webp) and videos (mp4, mov, avi, mkv, webm)"
        )
    
    story_data = StoryCreate(caption=caption, media_type=media_type)
    story = create_story(db, story_data, current_user.id, filename)
    
    story_dict = {**story.__dict__}
    story_dict["user"] = current_user
    
    return StoryResponse(**story_dict)

@router.delete("/expired")
async def cleanup_expired_stories(db: Session = Depends(get_db)):
    count = delete_expired_stories(db)
    return {"message": f"Deleted {count} expired stories"}