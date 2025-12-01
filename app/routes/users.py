from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
import os
import shutil

from database import get_db
from auth import get_current_active_user
from schemas import UserResponse, UserUpdate
from crud import get_user, update_user, follow_user, unfollow_user, get_followers, get_following
from models import User

router = APIRouter(prefix="/users", tags=["users"])

UPLOAD_DIR = "static/uploads/profile_pics"

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@router.get("/{user_id}", response_model=UserResponse)
async def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@router.put("/me", response_model=UserResponse)
async def update_user_me(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    return update_user(db, current_user.id, user_update)

@router.post("/me/profile-picture")
async def upload_profile_picture(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    
    # Delete old profile picture if not default
    if current_user.profile_picture and current_user.profile_picture != "default-profile.jpg":
        old_path = os.path.join(UPLOAD_DIR, current_user.profile_picture)
        if os.path.exists(old_path):
            os.remove(old_path)
    
    # Generate unique filename
    file_extension = os.path.splitext(file.filename)[1]
    filename = f"profile_{current_user.id}_{int(os.timestamp())}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Update user record
    update_data = UserUpdate(profile_picture=filename)
    updated_user = update_user(db, current_user.id, update_data)
    
    return {"filename": filename, "user": updated_user}

@router.post("/follow/{user_id}")
async def follow_user_route(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    follow = follow_user(db, current_user.id, user_id)
    if not follow:
        raise HTTPException(status_code=400, detail="Cannot follow yourself or already following")
    return {"message": "Followed successfully"}

@router.delete("/unfollow/{user_id}")
async def unfollow_user_route(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    if unfollow_user(db, current_user.id, user_id):
        return {"message": "Unfollowed successfully"}
    raise HTTPException(status_code=404, detail="Not following this user")

@router.get("/{user_id}/followers")
async def get_user_followers(user_id: int, db: Session = Depends(get_db)):
    followers = get_followers(db, user_id)
    return {"followers": [follower.follower for follower in followers]}

@router.get("/{user_id}/following")
async def get_user_following(user_id: int, db: Session = Depends(get_db)):
    following = get_following(db, user_id)
    return {"following": [follow.followed for follow in following]}