from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from auth import get_current_active_user
from schemas import ReactionCreate, ReactionResponse
from crud import create_reaction, delete_reaction
from models import User

router = APIRouter(prefix="/reactions", tags=["reactions"])

REACTION_TYPES = {
    "like": "👍",
    "love": "❤️",
    "laugh": "😂",
    "wow": "😮",
    "sad": "😢",
    "angry": "😠"
}

@router.post("/", response_model=ReactionResponse)
async def react_to_post(
    reaction: ReactionCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Validate reaction type
    if reaction.reaction_type not in REACTION_TYPES:
        raise HTTPException(status_code=400, detail="Invalid reaction type")
    
    # Set emoji if not provided
    if not reaction.emoji:
        reaction.emoji = REACTION_TYPES[reaction.reaction_type]
    
    db_reaction = create_reaction(db, reaction, current_user.id)
    
    reaction_dict = {**db_reaction.__dict__}
    reaction_dict["user"] = current_user
    
    return ReactionResponse(**reaction_dict)

@router.delete("/{post_id}")
async def remove_reaction(
    post_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    if delete_reaction(db, post_id, current_user.id):
        return {"message": "Reaction removed"}
    raise HTTPException(status_code=404, detail="Reaction not found")