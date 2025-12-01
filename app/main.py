from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
import os

from database import Base, engine, get_db
from auth import authenticate_user, create_access_token, get_current_active_user, ACCESS_TOKEN_EXPIRE_MINUTES
from schemas import UserCreate, UserLogin, Token
from crud import create_user, get_user_by_email
from routes import users, posts, stories, reactions, uploads
from websocket import websocket_endpoint
from models import User

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Social Media App",
    description="A professional social media platform",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(users.router)
app.include_router(posts.router)
app.include_router(stories.router)
app.include_router(reactions.router)
app.include_router(uploads.router)

# Authentication endpoints
@app.post("/register", response_model=Token)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    db_user = get_user_by_email(db, email=user_data.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user = create_user(db, user_data)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer", "user": user}

@app.post("/login", response_model=Token)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    user = authenticate_user(db, user_data.email, user_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer", "user": user}

# WebSocket endpoint
@app.websocket("/ws/{user_id}")
async def websocket_route(websocket: WebSocket, user_id: int):
    await websocket_endpoint(websocket, user_id)

# Serve HTML templates
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    with open("templates/login.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/register", response_class=HTMLResponse)
async def register_page():
    with open("templates/register.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/profile", response_class=HTMLResponse)
async def profile_page():
    with open("templates/profile.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/create", response_class=HTMLResponse)
async def create_post_page():
    with open("templates/create_post.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/stories", response_class=HTMLResponse)
async def stories_page():
    with open("templates/stories.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/messages", response_class=HTMLResponse)
async def messages_page():
    with open("templates/messages.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/settings", response_class=HTMLResponse)
async def settings_page():
    with open("templates/settings.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/explore", response_class=HTMLResponse)
async def explore_page():
    with open("templates/explore.html", "r") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)