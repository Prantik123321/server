from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import shutil
from datetime import datetime

from database import get_db
from auth import get_current_active_user
from schemas import PostCreate, PostResponse, CommentCreate, CommentResponse
from crud import (
    create_post, get_posts, get_user_posts, get_post, delete_post,
    create_comment, get_post_comments
)
from models import User, Post

router = APIRouter(prefix="/posts", tags=["posts"])

UPLOAD_DIR = "static/uploads/posts"
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

@router.get("/", response_model=List[PostResponse])
async def read_posts(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    db: Session = Depends(get_db)
):
    posts = get_posts(db, skip=skip, limit=limit)
    
    # Add counts to response
    response_posts = []
    for post in posts:
        post_dict = {**post.__dict__}
        post_dict["likes_count"] = len(post.reactions)
        post_dict["comments_count"] = len(post.comments)
        post_dict["owner"] = post.owner
        response_posts.append(PostResponse(**post_dict))
    
    return response_posts

@router.get("/user/{user_id}", response_model=List[PostResponse])
async def read_user_posts(user_id: int, db: Session = Depends(get_db)):
    posts = get_user_posts(db, user_id)
    
    response_posts = []
    for post in posts:
        post_dict = {**post.__dict__}
        post_dict["likes_count"] = len(post.reactions)
        post_dict["comments_count"] = len(post.comments)
        post_dict["owner"] = post.owner
        response_posts.append(PostResponse(**post_dict))
    
    return response_posts

@router.post("/", response_model=PostResponse)
async def create_new_post(
    caption: str = Form(None),
    file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    post_data = PostCreate(caption=caption)
    
    image_url = None
    video_url = None
    media_type = None
    
    if file:
        # Create upload directory if not exists
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)
        
        # Check file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        # Generate unique filename
        timestamp = int(datetime.utcnow().timestamp())
        filename = f"post_{current_user.id}_{timestamp}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Determine media type
        if file_extension in ALLOWED_IMAGE_EXTENSIONS:
            image_url = filename
            media_type = "image"
        elif file_extension in ALLOWED_VIDEO_EXTENSIONS:
            video_url = filename
            media_type = "video"
        else:
            raise HTTPException(
                status_code=400,
                detail="File type not allowed. Allowed types: images (jpg, png, gif, webp) and videos (mp4, mov, avi, mkv, webm)"
            )
    
    post = create_post(db, post_data, current_user.id, image_url, video_url)
    if media_type:
        post.media_type = media_type
        db.commit()
        db.refresh(post)
    
    post_dict = {**post.__dict__}
    post_dict["likes_count"] = 0
    post_dict["comments_count"] = 0
    post_dict["owner"] = current_user
    
    return PostResponse(**post_dict)

@router.get("/{post_id}", response_model=PostResponse)
async def read_post(post_id: int, db: Session = Depends(get_db)):
    post = get_post(db, post_id=post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    
    post_dict = {**post.__dict__}
    post_dict["likes_count"] = len(post.reactions)
    post_dict["comments_count"] = len(post.comments)
    post_dict["owner"] = post.owner
    
    return PostResponse(**post_dict)

@router.delete("/{post_id}")
async def delete_user_post(
    post_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    if delete_post(db, post_id, current_user.id):
        return {"message": "Post deleted successfully"}
    raise HTTPException(status_code=404, detail="Post not found or not authorized")

@router.post("/{post_id}/comments", response_model=CommentResponse)
async def add_comment(
    post_id: int,
    comment: CommentCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    comment.post_id = post_id
    db_comment = create_comment(db, comment, current_user.id)
    
    comment_dict = {**db_comment.__dict__}
    comment_dict["user"] = current_user
    
    return CommentResponse(**comment_dict)

@router.get("/{post_id}/comments", response_model=List[CommentResponse])
async def read_post_comments(post_id: int, db: Session = Depends(get_db)):
    comments = get_post_comments(db, post_id)
    
    response_comments = []
    for comment in comments:
        comment_dict = {**comment.__dict__}
        comment_dict["user"] = comment.user
        response_comments.append(CommentResponse(**comment_dict))
    
    return response_comments