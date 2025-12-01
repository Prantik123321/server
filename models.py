from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

db = SQLAlchemy()

class User(db.Model):
    """
    User model for storing user credentials and game statistics
    """
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.String(200), nullable=False)
    avatar = db.Column(db.String(200), default='default.png')
    
    # Game statistics
    total_wins = db.Column(db.Integer, default=0)
    total_losses = db.Column(db.Integer, default=0)
    total_draws = db.Column(db.Integer, default=0)
    total_games = db.Column(db.Integer, default=0)
    
    # Win rates
    rock_wins = db.Column(db.Integer, default=0)
    paper_wins = db.Column(db.Integer, default=0)
    scissors_wins = db.Column(db.Integer, default=0)
    
    # Streaks
    current_win_streak = db.Column(db.Integer, default=0)
    best_win_streak = db.Column(db.Integer, default=0)
    
    # Profile
    level = db.Column(db.Integer, default=1)
    experience = db.Column(db.Integer, default=0)
    coins = db.Column(db.Integer, default=100)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    last_played = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    leaderboard = db.relationship('Leaderboard', backref='user', lazy=True, uselist=False)
    games_as_player1 = db.relationship('Game', foreign_keys='Game.player1_id', backref='player1', lazy=True)
    games_as_player2 = db.relationship('Game', foreign_keys='Game.player2_id', backref='player2', lazy=True)
    friendships = db.relationship('Friendship', foreign_keys='Friendship.user_id', backref='user', lazy=True)
    friend_of = db.relationship('Friendship', foreign_keys='Friendship.friend_id', backref='friend', lazy=True)
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify password"""
        return check_password_hash(self.password_hash, password)
    
    def get_stats(self):
        """Get user statistics as dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'avatar': self.avatar,
            'total_wins': self.total_wins,
            'total_losses': self.total_losses,
            'total_draws': self.total_draws,
            'total_games': self.total_games,
            'win_rate': self.calculate_win_rate(),
            'rock_wins': self.rock_wins,
            'paper_wins': self.paper_wins,
            'scissors_wins': self.scissors_wins,
            'current_win_streak': self.current_win_streak,
            'best_win_streak': self.best_win_streak,
            'level': self.level,
            'experience': self.experience,
            'coins': self.coins,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def calculate_win_rate(self):
        """Calculate win rate percentage"""
        if self.total_games == 0:
            return 0
        return round((self.total_wins / self.total_games) * 100, 2)
    
    def add_experience(self, amount):
        """Add experience and check level up"""
        self.experience += amount
        exp_needed = self.level * 100
        
        while self.experience >= exp_needed:
            self.experience -= exp_needed
            self.level += 1
            exp_needed = self.level * 100
            self.coins += self.level * 50  # Reward coins on level up
    
    def update_win_streak(self, won):
        """Update win streak"""
        if won:
            self.current_win_streak += 1
            if self.current_win_streak > self.best_win_streak:
                self.best_win_streak = self.current_win_streak
        else:
            self.current_win_streak = 0
    
    def update_move_stats(self, move, won):
        """Update statistics for specific move"""
        if move == 'rock' and won:
            self.rock_wins += 1
        elif move == 'paper' and won:
            self.paper_wins += 1
        elif move == 'scissors' and won:
            self.scissors_wins += 1
    
    def __repr__(self):
        return f'<User {self.username}>'


class Leaderboard(db.Model):
    """
    Leaderboard model for ranking players
    """
    __tablename__ = 'leaderboard'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True, nullable=False)
    
    # Ranking scores
    score = db.Column(db.Integer, default=1000)  # Elo-like rating
    rank = db.Column(db.Integer, default=0)
    
    # Weekly/Monthly stats
    weekly_wins = db.Column(db.Integer, default=0)
    monthly_wins = db.Column(db.Integer, default=0)
    
    # Last updated
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def update_score(self, opponent_score, result):
        """
        Update Elo rating
        result: 'win', 'loss', or 'draw'
        """
        K = 32  # K-factor
        
        expected = 1 / (1 + 10 ** ((opponent_score - self.score) / 400))
        
        if result == 'win':
            actual = 1
        elif result == 'loss':
            actual = 0
        else:  # draw
            actual = 0.5
        
        self.score = round(self.score + K * (actual - expected))
        
        if result == 'win':
            self.weekly_wins += 1
            self.monthly_wins += 1
        
        self.last_updated = datetime.utcnow()
    
    def reset_weekly_stats(self):
        """Reset weekly statistics"""
        self.weekly_wins = 0
    
    def reset_monthly_stats(self):
        """Reset monthly statistics"""
        self.monthly_wins = 0
    
    def get_ranking_data(self):
        """Get data for leaderboard display"""
        return {
            'user_id': self.user_id,
            'username': self.user.username if self.user else 'Unknown',
            'avatar': self.user.avatar if self.user else 'default.png',
            'score': self.score,
            'rank': self.rank,
            'total_wins': self.user.total_wins if self.user else 0,
            'win_rate': self.user.calculate_win_rate() if self.user else 0,
            'level': self.user.level if self.user else 1
        }


class Game(db.Model):
    """
    Game model for storing completed game sessions
    """
    __tablename__ = 'games'
    
    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    
    # Players
    player1_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    player2_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Moves
    player1_move = db.Column(db.String(20), nullable=True)  # 'rock', 'paper', 'scissors'
    player2_move = db.Column(db.String(20), nullable=True)
    
    # Results
    winner_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    result = db.Column(db.String(20), nullable=True)  # 'player1_win', 'player2_win', 'draw'
    
    # Game details
    rounds = db.Column(db.Integer, default=1)
    player1_score = db.Column(db.Integer, default=0)
    player2_score = db.Column(db.Integer, default=0)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime, nullable=True)
    ended_at = db.Column(db.DateTime, nullable=True)
    
    # Game type
    game_type = db.Column(db.String(20), default='quick_match')  # 'quick_match', 'ranked', 'friendly'
    is_ranked = db.Column(db.Boolean, default=False)
    
    # Relationships
    winner = db.relationship('User', foreign_keys=[winner_id], backref='games_won')
    
    def get_game_data(self):
        """Get complete game data as dictionary"""
        return {
            'game_id': self.game_id,
            'player1': {
                'id': self.player1_id,
                'username': self.player1.username if self.player1 else 'Unknown',
                'move': self.player1_move,
                'score': self.player1_score
            },
            'player2': {
                'id': self.player2_id,
                'username': self.player2.username if self.player2 else 'Unknown',
                'move': self.player2_move,
                'score': self.player2_score
            },
            'winner_id': self.winner_id,
            'result': self.result,
            'rounds': self.rounds,
            'game_type': self.game_type,
            'is_ranked': self.is_ranked,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'duration': self.calculate_duration()
        }
    
    def calculate_duration(self):
        """Calculate game duration in seconds"""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return 0
    
    def determine_result(self):
        """Determine game result based on moves"""
        if not self.player1_move or not self.player2_move:
            return None
        
        moves = {'rock': 0, 'paper': 1, 'scissors': 2}
        
        p1 = moves.get(self.player1_move)
        p2 = moves.get(self.player2_move)
        
        if p1 is None or p2 is None:
            return None
        
        # Rock-paper-scissors logic
        if p1 == p2:
            self.result = 'draw'
            self.winner_id = None
        elif (p1 - p2) % 3 == 1:
            self.result = 'player1_win'
            self.winner_id = self.player1_id
        else:
            self.result = 'player2_win'
            self.winner_id = self.player2_id
        
        return self.result
    
    def update_player_stats(self):
        """Update player statistics based on game result"""
        if not self.result:
            return
        
        # Update player 1 stats
        player1 = User.query.get(self.player1_id)
        if player1:
            player1.total_games += 1
            if self.result == 'player1_win':
                player1.total_wins += 1
                player1.update_win_streak(True)
                player1.update_move_stats(self.player1_move, True)
                player1.add_experience(50)
                player1.coins += 20
            elif self.result == 'player2_win':
                player1.total_losses += 1
                player1.update_win_streak(False)
            else:  # draw
                player1.total_draws += 1
                player1.add_experience(10)
                player1.coins += 5
            
            player1.last_played = datetime.utcnow()
        
        # Update player 2 stats
        player2 = User.query.get(self.player2_id)
        if player2:
            player2.total_games += 1
            if self.result == 'player2_win':
                player2.total_wins += 1
                player2.update_win_streak(True)
                player2.update_move_stats(self.player2_move, True)
                player2.add_experience(50)
                player2.coins += 20
            elif self.result == 'player1_win':
                player2.total_losses += 1
                player2.update_win_streak(False)
            else:  # draw
                player2.total_draws += 1
                player2.add_experience(10)
                player2.coins += 5
            
            player2.last_played = datetime.utcnow()
        
        # Update leaderboard if ranked game
        if self.is_ranked:
            leader1 = Leaderboard.query.filter_by(user_id=self.player1_id).first()
            leader2 = Leaderboard.query.filter_by(user_id=self.player2_id).first()
            
            if leader1 and leader2:
                if self.result == 'player1_win':
                    leader1.update_score(leader2.score, 'win')
                    leader2.update_score(leader1.score, 'loss')
                elif self.result == 'player2_win':
                    leader1.update_score(leader2.score, 'loss')
                    leader2.update_score(leader1.score, 'win')
                else:  # draw
                    leader1.update_score(leader2.score, 'draw')
                    leader2.update_score(leader1.score, 'draw')
    
    def __repr__(self):
        return f'<Game {self.game_id}: {self.player1_id} vs {self.player2_id}>'


class Friendship(db.Model):
    """
    Friendship model for managing user friendships
    """
    __tablename__ = 'friendships'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    friend_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Friendship status
    status = db.Column(db.String(20), default='pending')  # 'pending', 'accepted', 'blocked'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    accepted_at = db.Column(db.DateTime, nullable=True)
    
    # Last interaction
    last_game_together = db.Column(db.DateTime, nullable=True)
    games_played_together = db.Column(db.Integer, default=0)
    
    # Unique constraint
    __table_args__ = (db.UniqueConstraint('user_id', 'friend_id', name='unique_friendship'),)
    
    def accept_friendship(self):
        """Accept friendship request"""
        self.status = 'accepted'
        self.accepted_at = datetime.utcnow()
    
    def block_friend(self):
        """Block a friend"""
        self.status = 'blocked'
    
    def get_friend_info(self):
        """Get friend information"""
        friend = User.query.get(self.friend_id)
        if not friend:
            return None
        
        return {
            'friend_id': friend.id,
            'username': friend.username,
            'avatar': friend.avatar,
            'status': friend.status if hasattr(friend, 'status') else 'offline',
            'level': friend.level,
            'friendship_status': self.status,
            'friends_since': self.accepted_at.isoformat() if self.accepted_at else None,
            'games_played_together': self.games_played_together,
            'last_game_together': self.last_game_together.isoformat() if self.last_game_together else None
        }


class ChatMessage(db.Model):
    """
    Chat message model for storing chat messages
    """
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Sender and receiver
    sender_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Message content
    message = db.Column(db.Text, nullable=False)
    message_type = db.Column(db.String(20), default='text')  # 'text', 'emoji', 'game_invite'
    
    # Status
    is_read = db.Column(db.Boolean, default=False)
    is_delivered = db.Column(db.Boolean, default=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    read_at = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    sender = db.relationship('User', foreign_keys=[sender_id], backref='sent_messages')
    receiver = db.relationship('User', foreign_keys=[receiver_id], backref='received_messages')
    
    def get_message_data(self):
        """Get message data as dictionary"""
        return {
            'message_id': self.id,
            'sender': {
                'id': self.sender_id,
                'username': self.sender.username if self.sender else 'Unknown',
                'avatar': self.sender.avatar if self.sender else 'default.png'
            },
            'receiver_id': self.receiver_id,
            'message': self.message,
            'message_type': self.message_type,
            'is_read': self.is_read,
            'is_delivered': self.is_delivered,
            'created_at': self.created_at.isoformat(),
            'read_at': self.read_at.isoformat() if self.read_at else None,
            'timestamp': self.created_at.timestamp()
        }


class Achievement(db.Model):
    """
    Achievement model for storing user achievements
    """
    __tablename__ = 'achievements'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    icon = db.Column(db.String(200), default='achievement_default.png')
    
    # Achievement requirements
    requirement_type = db.Column(db.String(50), nullable=False)  # 'total_wins', 'streak', 'move_wins', etc.
    requirement_value = db.Column(db.Integer, nullable=False)
    
    # Reward
    reward_coins = db.Column(db.Integer, default=0)
    reward_experience = db.Column(db.Integer, default=0)
    
    # Rarity
    rarity = db.Column(db.String(20), default='common')  # 'common', 'rare', 'epic', 'legendary'
    
    def check_achievement(self, user):
        """Check if user meets achievement requirements"""
        if self.requirement_type == 'total_wins':
            return user.total_wins >= self.requirement_value
        elif self.requirement_type == 'win_streak':
            return user.best_win_streak >= self.requirement_value
        elif self.requirement_type == 'rock_wins':
            return user.rock_wins >= self.requirement_value
        elif self.requirement_type == 'paper_wins':
            return user.paper_wins >= self.requirement_value
        elif self.requirement_type == 'scissors_wins':
            return user.scissors_wins >= self.requirement_value
        elif self.requirement_type == 'total_games':
            return user.total_games >= self.requirement_value
        elif self.requirement_type == 'level':
            return user.level >= self.requirement_value
        return False
    
    def get_achievement_data(self):
        """Get achievement data as dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'icon': self.icon,
            'requirement': {
                'type': self.requirement_type,
                'value': self.requirement_value
            },
            'reward': {
                'coins': self.reward_coins,
                'experience': self.reward_experience
            },
            'rarity': self.rarity
        }


class UserAchievement(db.Model):
    """
    User-Achievement relationship model
    """
    __tablename__ = 'user_achievements'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    achievement_id = db.Column(db.Integer, db.ForeignKey('achievements.id'), nullable=False)
    
    # Progress (for progressive achievements)
    progress = db.Column(db.Integer, default=0)
    is_unlocked = db.Column(db.Boolean, default=False)
    unlocked_at = db.Column(db.DateTime, nullable=True)
    
    # Unique constraint
    __table_args__ = (db.UniqueConstraint('user_id', 'achievement_id', name='unique_user_achievement'),)
    
    # Relationships
    user = db.relationship('User', backref='user_achievements')
    achievement = db.relationship('Achievement', backref='user_achievements')
    
    def unlock(self):
        """Unlock achievement for user"""
        self.is_unlocked = True
        self.unlocked_at = datetime.utcnow()
        
        # Give rewards
        if self.user and self.achievement:
            self.user.coins += self.achievement.reward_coins
            self.user.add_experience(self.achievement.reward_experience)


class Notification(db.Model):
    """
    Notification model for storing user notifications
    """
    __tablename__ = 'notifications'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Notification details
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    notification_type = db.Column(db.String(50), default='info')  # 'info', 'friend_request', 'game_invite', 'achievement'
    
    # Reference to related object
    related_id = db.Column(db.Integer, nullable=True)  # Can reference game_id, friend_id, etc.
    related_type = db.Column(db.String(50), nullable=True)  # 'game', 'friend', 'achievement'
    
    # Status
    is_read = db.Column(db.Boolean, default=False)
    is_archived = db.Column(db.Boolean, default=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    read_at = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    user = db.relationship('User', backref='notifications')
    
    def get_notification_data(self):
        """Get notification data as dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'message': self.message,
            'type': self.notification_type,
            'related': {
                'id': self.related_id,
                'type': self.related_type
            } if self.related_id else None,
            'is_read': self.is_read,
            'is_archived': self.is_archived,
            'created_at': self.created_at.isoformat(),
            'read_at': self.read_at.isoformat() if self.read_at else None
        }


# Helper functions
def create_default_achievements():
    """Create default achievements"""
    default_achievements = [
        {
            'name': 'First Win',
            'description': 'Win your first game',
            'icon': 'first_win.png',
            'requirement_type': 'total_wins',
            'requirement_value': 1,
            'reward_coins': 100,
            'reward_experience': 50,
            'rarity': 'common'
        },
        {
            'name': 'Rock Master',
            'description': 'Win 50 games with Rock',
            'icon': 'rock_master.png',
            'requirement_type': 'rock_wins',
            'requirement_value': 50,
            'reward_coins': 500,
            'reward_experience': 200,
            'rarity': 'rare'
        },
        {
            'name': 'Paper Champion',
            'description': 'Win 50 games with Paper',
            'icon': 'paper_champion.png',
            'requirement_type': 'paper_wins',
            'requirement_value': 50,
            'reward_coins': 500,
            'reward_experience': 200,
            'rarity': 'rare'
        },
        {
            'name': 'Scissors Specialist',
            'description': 'Win 50 games with Scissors',
            'icon': 'scissors_specialist.png',
            'requirement_type': 'scissors_wins',
            'requirement_value': 50,
            'reward_coins': 500,
            'reward_experience': 200,
            'rarity': 'rare'
        },
        {
            'name': 'Unstoppable',
            'description': 'Achieve a 10-game win streak',
            'icon': 'unstoppable.png',
            'requirement_type': 'win_streak',
            'requirement_value': 10,
            'reward_coins': 1000,
            'reward_experience': 500,
            'rarity': 'epic'
        },
        {
            'name': 'Veteran Player',
            'description': 'Play 100 games',
            'icon': 'veteran.png',
            'requirement_type': 'total_games',
            'requirement_value': 100,
            'reward_coins': 300,
            'reward_experience': 150,
            'rarity': 'common'
        },
        {
            'name': 'Level 10',
            'description': 'Reach level 10',
            'icon': 'level_10.png',
            'requirement_type': 'level',
            'requirement_value': 10,
            'reward_coins': 1000,
            'reward_experience': 500,
            'rarity': 'rare'
        }
    ]
    
    return default_achievements


def init_database():
    """Initialize database with default data"""
    # Create tables
    db.create_all()
    
    # Create default achievements
    for achievement_data in create_default_achievements():
        achievement = Achievement.query.filter_by(name=achievement_data['name']).first()
        if not achievement:
            achievement = Achievement(**achievement_data)
            db.session.add(achievement)
    
    db.session.commit()
    print("Database initialized successfully!")