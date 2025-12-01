from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from typing import List, Optional
import os

from models import User, Post, Comment, Reaction, Story, Follow, Notification, Message
from schemas import (
    UserCreate, UserUpdate, PostCreate, CommentCreate, 
    ReactionCreate, StoryCreate, FollowBase, MessageBase
)
from auth import get_password_hash

# User CRUD operations
def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(User).offset(skip).limit(limit).all()

def create_user(db: Session, user: UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user(db: Session, user_id: int, user_update: UserUpdate):
    db_user = get_user(db, user_id)
    if db_user:
        update_data = user_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_user, key, value)
        db.commit()
        db.refresh(db_user)
    return db_user

# Post CRUD operations
def create_post(db: Session, post: PostCreate, owner_id: int, image_url: str = None, video_url: str = None):
    db_post = Post(
        **post.dict(),
        owner_id=owner_id,
        image_url=image_url,
        video_url=video_url
    )
    db.add(db_post)
    db.commit()
    db.refresh(db_post)
    return db_post

def get_post(db: Session, post_id: int):
    return db.query(Post).filter(Post.id == post_id).first()

def get_posts(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Post).order_by(desc(Post.created_at)).offset(skip).limit(limit).all()

def get_user_posts(db: Session, user_id: int):
    return db.query(Post).filter(Post.owner_id == user_id).order_by(desc(Post.created_at)).all()

def delete_post(db: Session, post_id: int, user_id: int):
    post = db.query(Post).filter(Post.id == post_id, Post.owner_id == user_id).first()
    if post:
        db.delete(post)
        db.commit()
        return True
    return False

# Comment CRUD operations
def create_comment(db: Session, comment: CommentCreate, user_id: int):
    db_comment = Comment(**comment.dict(), user_id=user_id)
    db.add(db_comment)
    db.commit()
    db.refresh(db_comment)
    
    # Create notification for post owner
    post = get_post(db, comment.post_id)
    if post and post.owner_id != user_id:
        notification = Notification(
            user_id=post.owner_id,
            type="comment",
            message=f"{db_comment.user.username} commented on your post",
            data={"post_id": post.id, "comment_id": db_comment.id}
        )
        db.add(notification)
        db.commit()
    
    return db_comment

def get_post_comments(db: Session, post_id: int):
    return db.query(Comment).filter(Comment.post_id == post_id).order_by(Comment.created_at).all()

# Reaction CRUD operations
def create_reaction(db: Session, reaction: ReactionCreate, user_id: int):
    # Check if user already reacted
    existing = db.query(Reaction).filter(
        Reaction.post_id == reaction.post_id,
        Reaction.user_id == user_id
    ).first()
    
    if existing:
        # Update existing reaction
        existing.reaction_type = reaction.reaction_type
        existing.emoji = reaction.emoji
        db.commit()
        db.refresh(existing)
        
        # Create notification for post owner
        post = get_post(db, reaction.post_id)
        if post and post.owner_id != user_id:
            notification = Notification(
                user_id=post.owner_id,
                type="reaction",
                message=f"{existing.user.username} reacted to your post",
                data={"post_id": post.id, "reaction_type": reaction.reaction_type}
            )
            db.add(notification)
            db.commit()
        
        return existing
    
    db_reaction = Reaction(**reaction.dict(), user_id=user_id)
    db.add(db_reaction)
    db.commit()
    db.refresh(db_reaction)
    
    # Create notification for post owner
    post = get_post(db, reaction.post_id)
    if post and post.owner_id != user_id:
        notification = Notification(
            user_id=post.owner_id,
            type="reaction",
            message=f"{db_reaction.user.username} reacted to your post",
            data={"post_id": post.id, "reaction_type": reaction.reaction_type}
        )
        db.add(notification)
        db.commit()
    
    return db_reaction

def delete_reaction(db: Session, post_id: int, user_id: int):
    reaction = db.query(Reaction).filter(
        Reaction.post_id == post_id,
        Reaction.user_id == user_id
    ).first()
    if reaction:
        db.delete(reaction)
        db.commit()
        return True
    return False

# Story CRUD operations
def create_story(db: Session, story: StoryCreate, user_id: int, media_url: str):
    expires_at = datetime.utcnow() + timedelta(hours=24)
    db_story = Story(
        **story.dict(),
        user_id=user_id,
        media_url=media_url,
        expires_at=expires_at
    )
    db.add(db_story)
    db.commit()
    db.refresh(db_story)
    return db_story

def get_active_stories(db: Session, user_id: Optional[int] = None):
    now = datetime.utcnow()
    query = db.query(Story).filter(Story.expires_at > now)
    if user_id:
        query = query.filter(Story.user_id == user_id)
    return query.order_by(desc(Story.created_at)).all()

def delete_expired_stories(db: Session):
    now = datetime.utcnow()
    expired_stories = db.query(Story).filter(Story.expires_at <= now).all()
    for story in expired_stories:
        # Delete the file from storage
        if os.path.exists(story.media_url):
            os.remove(story.media_url)
        db.delete(story)
    db.commit()
    return len(expired_stories)

# Follow CRUD operations
def follow_user(db: Session, follower_id: int, followed_id: int):
    if follower_id == followed_id:
        return None
    
    existing = db.query(Follow).filter(
        Follow.follower_id == follower_id,
        Follow.followed_id == followed_id
    ).first()
    
    if existing:
        return existing
    
    db_follow = Follow(follower_id=follower_id, followed_id=followed_id)
    db.add(db_follow)
    db.commit()
    db.refresh(db_follow)
    
    # Create notification
    notification = Notification(
        user_id=followed_id,
        type="follow",
        message=f"{db_follow.follower.username} started following you",
        data={"follower_id": follower_id}
    )
    db.add(notification)
    db.commit()
    
    return db_follow

def unfollow_user(db: Session, follower_id: int, followed_id: int):
    follow = db.query(Follow).filter(
        Follow.follower_id == follower_id,
        Follow.followed_id == followed_id
    ).first()
    if follow:
        db.delete(follow)
        db.commit()
        return True
    return False

def get_followers(db: Session, user_id: int):
    return db.query(Follow).filter(Follow.followed_id == user_id).all()

def get_following(db: Session, user_id: int):
    return db.query(Follow).filter(Follow.follower_id == user_id).all()

def is_following(db: Session, follower_id: int, followed_id: int):
    return db.query(Follow).filter(
        Follow.follower_id == follower_id,
        Follow.followed_id == followed_id
    ).first() is not None

# Notification CRUD operations
def get_user_notifications(db: Session, user_id: int, unread_only: bool = False):
    query = db.query(Notification).filter(Notification.user_id == user_id)
    if unread_only:
        query = query.filter(Notification.is_read == False)
    return query.order_by(desc(Notification.created_at)).all()

def mark_notification_read(db: Session, notification_id: int, user_id: int):
    notification = db.query(Notification).filter(
        Notification.id == notification_id,
        Notification.user_id == user_id
    ).first()
    if notification:
        notification.is_read = True
        db.commit()
        db.refresh(notification)
    return notification

def mark_all_notifications_read(db: Session, user_id: int):
    notifications = db.query(Notification).filter(
        Notification.user_id == user_id,
        Notification.is_read == False
    ).all()
    for notification in notifications:
        notification.is_read = True
    db.commit()
    return len(notifications)

# Message CRUD operations
def create_message(db: Session, message: MessageBase, sender_id: int):
    db_message = Message(**message.dict(), sender_id=sender_id)
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    
    # Create notification for receiver
    notification = Notification(
        user_id=message.receiver_id,
        type="message",
        message=f"You have a new message from {db_message.sender.username}",
        data={"message_id": db_message.id, "sender_id": sender_id}
    )
    db.add(notification)
    db.commit()
    
    return db_message

def get_conversation(db: Session, user1_id: int, user2_id: int):
    return db.query(Message).filter(
        ((Message.sender_id == user1_id) & (Message.receiver_id == user2_id)) |
        ((Message.sender_id == user2_id) & (Message.receiver_id == user1_id))
    ).order_by(Message.created_at).all()

def get_user_conversations(db: Session, user_id: int):
    # Get distinct users that the current user has conversed with
    sent = db.query(Message.receiver_id).filter(Message.sender_id == user_id).distinct()
    received = db.query(Message.sender_id).filter(Message.receiver_id == user_id).distinct()
    user_ids = set([id[0] for id in sent] + [id[0] for id in received])
    
    conversations = []
    for other_user_id in user_ids:
        last_message = db.query(Message).filter(
            ((Message.sender_id == user_id) & (Message.receiver_id == other_user_id)) |
            ((Message.sender_id == other_user_id) & (Message.receiver_id == user_id))
        ).order_by(desc(Message.created_at)).first()
        
        if last_message:
            other_user = get_user(db, other_user_id)
            unread_count = db.query(Message).filter(
                Message.sender_id == other_user_id,
                Message.receiver_id == user_id,
                Message.is_read == False
            ).count()
            
            conversations.append({
                "user": other_user,
                "last_message": last_message,
                "unread_count": unread_count
            })
    
    return sorted(conversations, key=lambda x: x["last_message"].created_at, reverse=True)

def mark_messages_read(db: Session, sender_id: int, receiver_id: int):
    messages = db.query(Message).filter(
        Message.sender_id == sender_id,
        Message.receiver_id == receiver_id,
        Message.is_read == False
    ).all()
    for message in messages:
        message.is_read = True
    db.commit()
    return len(messages)