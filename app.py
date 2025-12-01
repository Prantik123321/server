from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
import random

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///rps_game.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-this')

db = SQLAlchemy(app)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    wins = db.Column(db.Integer, default=0)
    losses = db.Column(db.Integer, default=0)
    draws = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Leaderboard(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    score = db.Column(db.Integer, default=0)
    wins = db.Column(db.Integer, default=0)
    losses = db.Column(db.Integer, default=0)
    user = db.relationship('User', backref=db.backref('leaderboard', lazy=True))

class GameSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    player1_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    player2_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    winner_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    status = db.Column(db.String(20), default='completed')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Game state management
active_games = {}
waiting_players = {}
user_socket_map = {}

# Routes
@app.route('/')
def home():
    return jsonify({
        "message": "Rock Paper Scissors Server",
        "status": "online",
        "players_waiting": len(waiting_players)
    })

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({"error": "Username and password required"}), 400
        
        if len(username) < 3:
            return jsonify({"error": "Username must be at least 3 characters"}), 400
        
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400
        
        if User.query.filter_by(username=username).first():
            return jsonify({"error": "Username already exists"}), 400
        
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        # Create leaderboard entry
        leaderboard = Leaderboard(user_id=user.id)
        db.session.add(leaderboard)
        db.session.commit()
        
        return jsonify({
            "message": "User created successfully", 
            "user_id": user.id,
            "username": user.username
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({"error": "Username and password required"}), 400
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            return jsonify({
                "message": "Login successful",
                "user_id": user.id,
                "username": user.username,
                "wins": user.wins,
                "losses": user.losses,
                "draws": user.draws
            })
        
        return jsonify({"error": "Invalid credentials"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/profile/<int:user_id>')
def get_profile(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify({
        "username": user.username,
        "wins": user.wins,
        "losses": user.losses,
        "draws": user.draws,
        "total_games": user.wins + user.losses + user.draws,
        "win_rate": round((user.wins / max(user.wins + user.losses + user.draws, 1)) * 100, 2)
    })

@app.route('/api/leaderboard')
def get_leaderboard():
    leaders = Leaderboard.query.join(User).order_by(Leaderboard.score.desc()).limit(20).all()
    result = []
    for idx, leader in enumerate(leaders, 1):
        result.append({
            'rank': idx,
            'username': leader.user.username,
            'score': leader.score,
            'wins': leader.wins,
            'losses': leader.losses
        })
    return jsonify(result)

@app.route('/api/stats')
def get_stats():
    total_users = User.query.count()
    total_games = GameSession.query.count()
    active_players = len(user_socket_map)
    
    return jsonify({
        "total_users": total_users,
        "total_games": total_games,
        "active_players": active_players,
        "games_in_progress": len(active_games)
    })

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    
    # Remove from waiting list
    if request.sid in waiting_players:
        user_data = waiting_players.pop(request.sid)
        print(f"Removed {user_data.get('username')} from waiting list")
    
    # Remove from user-socket map
    for user_id, sid in list(user_socket_map.items()):
        if sid == request.sid:
            user_socket_map.pop(user_id)
            break
    
    # Handle game disconnection
    for game_id in list(active_games.keys()):
        game = active_games[game_id]
        if request.sid == game['player1'] or request.sid == game['player2']:
            opponent_sid = game['player2'] if request.sid == game['player1'] else game['player1']
            emit('opponent_disconnected', {'message': 'Opponent disconnected'}, room=opponent_sid)
            
            # Update user stats
            try:
                if game.get('player1_user_id'):
                    user = User.query.get(game['player1_user_id'])
                    if user:
                        user.losses += 1
                        db.session.commit()
                
                if game.get('player2_user_id'):
                    user = User.query.get(game['player2_user_id'])
                    if user:
                        user.wins += 1
                        db.session.commit()
            except:
                pass
            
            # Clean up game
            leave_room(game_id, opponent_sid)
            active_games.pop(game_id, None)
            break

@socketio.on('authenticate')
def handle_authenticate(data):
    user_id = data.get('user_id')
    if user_id:
        user_socket_map[user_id] = request.sid
        emit('authenticated', {'message': 'Authentication successful'})

@socketio.on('find_match')
def handle_find_match(data):
    user_id = data.get('user_id')
    username = data.get('username')
    
    if not user_id or not username:
        emit('error', {'message': 'User information required'})
        return
    
    # Update socket mapping
    user_socket_map[user_id] = request.sid
    
    if request.sid in waiting_players:
        emit('message', {'message': 'Already looking for match'})
        return
    
    # Add to waiting list
    waiting_players[request.sid] = {
        'user_id': user_id,
        'username': username,
        'sid': request.sid
    }
    
    emit('searching', {'message': 'Looking for opponent...'})
    
    # Try to match players
    if len(waiting_players) >= 2:
        waiting_list = list(waiting_players.items())
        player1_sid, player1_data = waiting_list[0]
        player2_sid, player2_data = waiting_list[1]
        
        # Remove from waiting list
        waiting_players.pop(player1_sid, None)
        waiting_players.pop(player2_sid, None)
        
        # Create game
        game_id = f"game_{datetime.now().timestamp()}_{random.randint(1000, 9999)}"
        active_games[game_id] = {
            'player1': player1_sid,
            'player2': player2_sid,
            'player1_user_id': player1_data['user_id'],
            'player2_user_id': player2_data['user_id'],
            'player1_username': player1_data['username'],
            'player2_username': player2_data['username'],
            'choices': {},
            'scores': {player1_sid: 0, player2_sid: 0},
            'round': 1
        }
        
        # Join room
        join_room(game_id, player1_sid)
        join_room(game_id, player2_sid)
        
        # Notify players
        emit('match_found', {
            'game_id': game_id,
            'opponent': player2_data['username'],
            'your_side': 'player1'
        }, room=player1_sid)
        
        emit('match_found', {
            'game_id': game_id,
            'opponent': player1_data['username'],
            'your_side': 'player2'
        }, room=player2_sid)
        
        # Start game
        emit('game_start', {
            'message': 'Game started! Make your choice.',
            'round': 1
        }, room=game_id)

@socketio.on('make_choice')
def handle_choice(data):
    game_id = data.get('game_id')
    choice = data.get('choice')
    
    if not game_id or not choice:
        emit('error', {'message': 'Game ID and choice required'})
        return
    
    if game_id not in active_games:
        emit('error', {'message': 'Game not found'})
        return
    
    game = active_games[game_id]
    
    # Validate choice
    valid_choices = ['rock', 'paper', 'scissors']
    if choice not in valid_choices:
        emit('error', {'message': 'Invalid choice'})
        return
    
    # Record choice
    game['choices'][request.sid] = choice
    
    # Notify player
    emit('choice_received', {'choice': choice})
    
    # Notify opponent
    opponent_sid = game['player2'] if request.sid == game['player1'] else game['player1']
    emit('opponent_choosing', {}, room=opponent_sid)
    
    # Check if both made choices
    if len(game['choices']) == 2:
        determine_winner(game_id)

@socketio.on('play_again')
def handle_play_again(data):
    game_id = data.get('game_id')
    
    if game_id not in active_games:
        return
    
    game = active_games[game_id]
    
    # Reset choices
    game['choices'] = {}
    game['round'] += 1
    
    # Start new round
    emit('next_round', {
        'round': game['round'],
        'scores': game['scores']
    }, room=game_id)

@socketio.on('leave_game')
def handle_leave_game(data):
    game_id = data.get('game_id')
    
    if game_id in active_games:
        game = active_games[game_id]
        
        # Notify opponent
        opponent_sid = game['player2'] if request.sid == game['player1'] else game['player1']
        if opponent_sid:
            emit('opponent_left', {'message': 'Opponent left the game'}, room=opponent_sid)
        
        # Clean up
        leave_room(game_id, request.sid)
        if opponent_sid:
            leave_room(game_id, opponent_sid)
        
        active_games.pop(game_id, None)

def determine_winner(game_id):
    game = active_games[game_id]
    p1_sid = game['player1']
    p2_sid = game['player2']
    
    p1_choice = game['choices'][p1_sid]
    p2_choice = game['choices'][p2_sid]
    
    # Game logic
    if p1_choice == p2_choice:
        winner_sid = None
        result_text = "Draw!"
    elif (p1_choice == 'rock' and p2_choice == 'scissors') or \
         (p1_choice == 'paper' and p2_choice == 'rock') or \
         (p1_choice == 'scissors' and p2_choice == 'paper'):
        winner_sid = p1_sid
        game['scores'][p1_sid] += 1
        result_text = f"{game['player1_username']} wins!"
    else:
        winner_sid = p2_sid
        game['scores'][p2_sid] += 1
        result_text = f"{game['player2_username']} wins!"
    
    # Update user stats in database
    try:
        if winner_sid == p1_sid:
            user = User.query.get(game['player1_user_id'])
            if user:
                user.wins += 1
                
            user2 = User.query.get(game['player2_user_id'])
            if user2:
                user2.losses += 1
                
            # Update leaderboard
            leader1 = Leaderboard.query.filter_by(user_id=game['player1_user_id']).first()
            if leader1:
                leader1.score += 10
                leader1.wins += 1
                
        elif winner_sid == p2_sid:
            user = User.query.get(game['player2_user_id'])
            if user:
                user.wins += 1
                
            user1 = User.query.get(game['player1_user_id'])
            if user1:
                user1.losses += 1
                
            # Update leaderboard
            leader2 = Leaderboard.query.filter_by(user_id=game['player2_user_id']).first()
            if leader2:
                leader2.score += 10
                leader2.wins += 1
        else:
            # Draw
            user1 = User.query.get(game['player1_user_id'])
            if user1:
                user1.draws += 1
                
            user2 = User.query.get(game['player2_user_id'])
            if user2:
                user2.draws += 1
        
        db.session.commit()
        
        # Record game session
        game_session = GameSession(
            player1_id=game['player1_user_id'],
            player2_id=game['player2_user_id'],
            winner_id=game['player1_user_id'] if winner_sid == p1_sid else game['player2_user_id'] if winner_sid else None,
            status='completed'
        )
        db.session.add(game_session)
        db.session.commit()
        
    except Exception as e:
        print(f"Error updating stats: {e}")
    
    # Send results to players
    result_data = {
        'player1_choice': p1_choice,
        'player2_choice': p2_choice,
        'result': 'win' if winner_sid == p1_sid else 'lose' if winner_sid == p2_sid else 'draw',
        'result_text': result_text,
        'scores': {
            game['player1_username']: game['scores'][p1_sid],
            game['player2_username']: game['scores'][p2_sid]
        },
        'winner': game['player1_username'] if winner_sid == p1_sid else game['player2_username'] if winner_sid else None
    }
    
    emit('game_result', result_data, room=game_id)
    
    # Clear choices for next round
    game['choices'] = {}
    
    # Check if game over (best of 3)
    if game['scores'][p1_sid] >= 2 or game['scores'][p2_sid] >= 2:
        game_winner = game['player1_username'] if game['scores'][p1_sid] >= 2 else game['player2_username']
        emit('game_over', {
            'winner': game_winner,
            'final_scores': result_data['scores']
        }, room=game_id)
        
        # Clean up after delay
        @socketio.sleep(5)
        def cleanup():
            if game_id in active_games:
                leave_room(game_id, p1_sid)
                leave_room(game_id, p2_sid)
                active_games.pop(game_id, None)
        
        socketio.start_background_task(cleanup)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
from models import db, User, Game, Leaderboard, Friendship, ChatMessage, Achievement, UserAchievement, Notification, init_database

# app creation code...

if __name__ == '__main__':
    with app.app_context():
        # Initialize database
        init_database()
    socketio.run(app, host='0.0.0.0', port=port, debug=True)