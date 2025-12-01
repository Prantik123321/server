from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List

# User schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    bio: Optional[str] = None
    profile_picture: Optional[str] = None
    dark_mode: Optional[bool] = None

class UserResponse(UserBase):
    id: int
    profile_picture: str
    bio: Optional[str]
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

# Post schemas
class PostBase(BaseModel):
    caption: Optional[str] = None
    media_type: Optional[str] = None

class PostCreate(PostBase):
    pass

class PostResponse(PostBase):
    id: int
    owner_id: int
    image_url: Optional[str]
    video_url: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    likes_count: int = 0
    comments_count: int = 0
    owner: UserResponse
    
    class Config:
        from_attributes = True

# Comment schemas
class CommentBase(BaseModel):
    content: str

class CommentCreate(CommentBase):
    post_id: int

class CommentResponse(CommentBase):
    id: int
    post_id: int
    user_id: int
    created_at: datetime
    user: UserResponse
    
    class Config:
        from_attributes = True

# Reaction schemas
class ReactionBase(BaseModel):
    reaction_type: str
    emoji: Optional[str] = None

class ReactionCreate(ReactionBase):
    post_id: int

class ReactionResponse(ReactionBase):
    id: int
    post_id: int
    user_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Story schemas
class StoryBase(BaseModel):
    caption: Optional[str] = None

class StoryCreate(StoryBase):
    media_type: str

class StoryResponse(StoryBase):
    id: int
    user_id: int
    media_url: str
    media_type: str
    created_at: datetime
    expires_at: datetime
    user: UserResponse
    
    class Config:
        from_attributes = True

# Follow schemas
class FollowBase(BaseModel):
    followed_id: int

class FollowResponse(BaseModel):
    follower_id: int
    followed_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Notification schemas
class NotificationResponse(BaseModel):
    id: int
    type: str
    message: str
    data: Optional[dict]
    is_read: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

# Message schemas
class MessageBase(BaseModel):
    content: str
    receiver_id: int

class MessageResponse(BaseModel):
    id: int
    sender_id: int
    receiver_id: int
    content: str
    is_read: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

# Token schemas
class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class TokenData(BaseModel):
    username: Optional[str] = None