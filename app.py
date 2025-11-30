from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
from models import db, User, GameSession, ChatMessage, Leaderboard
import random
import time
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///game.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# In-memory storage for active games and players
active_games = {}
waiting_players = {}
connected_users = {}

# Authentication routes
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')

        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists'}), 400

        user = User(username=username, email=email)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'User created successfully',
            'user': user.to_dict()
        }), 201

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        user = User.query.filter_by(username=username).first()
        
        if not user or not user.check_password(password):
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

        # Update user status
        user.is_online = True
        user.last_seen = datetime.utcnow()
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': user.to_dict()
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/profile/<int:user_id>', methods=['GET'])
def get_profile(user_id):
    try:
        user = User.query.get_or_404(user_id)
        return jsonify({'success': True, 'user': user.to_dict()}), 200
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    try:
        # Get top 50 players by score
        top_players = User.query.order_by(User.total_score.desc()).limit(50).all()
        
        leaderboard_data = []
        for rank, user in enumerate(top_players, 1):
            user_data = user.to_dict()
            user_data['rank'] = rank
            leaderboard_data.append(user_data)
        
        return jsonify({
            'success': True,
            'leaderboard': leaderboard_data
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    
    # Clean up user data
    user_id = connected_users.get(request.sid)
    if user_id:
        user = User.query.get(user_id)
        if user:
            user.is_online = False
            user.last_seen = datetime.utcnow()
            db.session.commit()
        
        del connected_users[request.sid]
    
    # Remove from waiting players
    if request.sid in waiting_players:
        del waiting_players[request.sid]
    
    # Clean up games
    for game_id, game in list(active_games.items()):
        if request.sid in [game['player1_sid'], game['player2_sid']]:
            handle_game_disconnect(game_id, request.sid)

@socketio.on('user_authenticated')
def handle_user_authenticated(data):
    user_id = data.get('user_id')
    connected_users[request.sid] = user_id
    
    # Update user status
    user = User.query.get(user_id)
    if user:
        user.is_online = True
        user.last_seen = datetime.utcnow()
        db.session.commit()
    
    emit('authentication_confirmed', {'success': True})

@socketio.on('find_match')
def handle_find_match(data):
    user_id = data.get('user_id')
    username = data.get('username')
    
    if not user_id or not username:
        emit('error', {'message': 'User information required'})
        return
    
    # Add to waiting players
    waiting_players[request.sid] = {
        'user_id': user_id,
        'username': username,
        'joined_at': time.time()
    }
    
    emit('searching_match', {'message': 'Searching for opponent...'})
    
    # Try to match with another player
    try_match_players()

@socketio.on('cancel_matchmaking')
def handle_cancel_matchmaking():
    if request.sid in waiting_players:
        del waiting_players[request.sid]
        emit('matchmaking_cancelled', {'message': 'Matchmaking cancelled'})

@socketio.on('send_chat_message')
def handle_chat_message(data):
    room_id = data.get('room_id')
    user_id = data.get('user_id')
    username = data.get('username')
    message = data.get('message')
    
    if not all([room_id, user_id, username, message]):
        return
    
    # Save chat message to database
    chat_message = ChatMessage(
        room_id=room_id,
        user_id=user_id,
        username=username,
        message=message
    )
    db.session.add(chat_message)
    db.session.commit()
    
    # Broadcast to room
    emit('new_chat_message', {
        'username': username,
        'message': message,
        'timestamp': datetime.utcnow().isoformat()
    }, room=room_id)

@socketio.on('make_choice')
def handle_choice(data):
    game_id = data.get('game_id')
    choice = data.get('choice')
    user_id = data.get('user_id')
    
    if game_id not in active_games:
        emit('error', {'message': 'Game not found'})
        return
    
    game = active_games[game_id]
    
    # Update player choice
    if game['player1_sid'] == request.sid:
        game['player1_choice'] = choice
        game['player1_ready'] = True
    elif game['player2_sid'] == request.sid:
        game['player2_choice'] = choice
        game['player2_ready'] = True
    
    # Notify players about choice made
    emit('player_choice_made', {
        'player': 'player1' if game['player1_sid'] == request.sid else 'player2'
    }, room=game_id)
    
    # Check if both players made their choices
    if game['player1_ready'] and game['player2_ready']:
        calculate_round_result(game_id)

@socketio.on('join_game_room')
def handle_join_game_room(data):
    game_id = data.get('game_id')
    join_room(game_id)
    emit('joined_room', {'game_id': game_id})

@socketio.on('leave_game_room')
def handle_leave_game_room(data):
    game_id = data.get('game_id')
    leave_room(game_id)
    emit('left_room', {'game_id': game_id})

def try_match_players():
    if len(waiting_players) < 2:
        return
    
    # Get two waiting players
    player_sids = list(waiting_players.keys())[:2]
    player1_data = waiting_players[player_sids[0]]
    player2_data = waiting_players[player_sids[1]]
    
    # Create game room
    game_id = f"game_{int(time.time())}_{random.randint(1000, 9999)}"
    
    active_games[game_id] = {
        'player1_sid': player_sids[0],
        'player2_sid': player_sids[1],
        'player1_id': player1_data['user_id'],
        'player2_id': player2_data['user_id'],
        'player1_username': player1_data['username'],
        'player2_username': player2_data['username'],
        'player1_choice': None,
        'player2_choice': None,
        'player1_ready': False,
        'player2_ready': False,
        'player1_score': 0,
        'player2_score': 0,
        'current_round': 1,
        'max_rounds': 3,
        'created_at': time.time()
    }
    
    # Remove from waiting list
    del waiting_players[player_sids[0]]
    del waiting_players[player_sids[1]]
    
    # Notify players
    emit('match_found', {
        'game_id': game_id,
        'opponent': player2_data['username'],
        'player_number': 1
    }, room=player_sids[0])
    
    emit('match_found', {
        'game_id': game_id,
        'opponent': player1_data['username'],
        'player_number': 2
    }, room=player_sids[1])
    
    # Create game session in database
    game_session = GameSession(
        room_id=game_id,
        player1_id=player1_data['user_id'],
        player2_id=player2_data['user_id'],
        status='playing'
    )
    db.session.add(game_session)
    db.session.commit()
    
    # Join both players to game room
    join_room(game_id, sid=player_sids[0])
    join_room(game_id, sid=player_sids[1])
    
    # Start first round
    emit('round_start', {
        'round_number': 1,
        'max_rounds': 3
    }, room=game_id)

def calculate_round_result(game_id):
    game = active_games[game_id]
    p1_choice = game['player1_choice']
    p2_choice = game['player2_choice']
    
    # Determine winner
    if p1_choice == p2_choice:
        result = 'draw'
        winner = None
    elif (p1_choice == 'rock' and p2_choice == 'scissors') or \
         (p1_choice == 'paper' and p2_choice == 'rock') or \
         (p1_choice == 'scissors' and p2_choice == 'paper'):
        result = 'player1_win'
        winner = game['player1_username']
        game['player1_score'] += 1
    else:
        result = 'player2_win'
        winner = game['player2_username']
        game['player2_score'] += 1
    
    # Update database with game result
    game_session = GameSession.query.filter_by(room_id=game_id).first()
    if game_session:
        game_session.player1_score = game['player1_score']
        game_session.player2_score = game['player2_score']
        db.session.commit()
    
    # Send round result
    emit('round_result', {
        'result': result,
        'winner': winner,
        'player1_choice': p1_choice,
        'player2_choice': p2_choice,
        'player1_score': game['player1_score'],
        'player2_score': game['player2_score'],
        'current_round': game['current_round']
    }, room=game_id)
    
    # Reset choices for next round
    game['player1_choice'] = None
    game['player2_choice'] = None
    game['player1_ready'] = False
    game['player2_ready'] = False
    
    # Check if game should continue
    game['current_round'] += 1
    
    max_score = max(game['player1_score'], game['player2_score'])
    rounds_left = game['max_rounds'] - game['current_round'] + 1
    
    if max_score > rounds_left or game['current_round'] > game['max_rounds']:
        # Game over
        end_game(game_id)
    else:
        # Next round
        emit('next_round', {
            'round_number': game['current_round']
        }, room=game_id)

def end_game(game_id):
    game = active_games[game_id]
    
    # Determine final winner
    if game['player1_score'] > game['player2_score']:
        final_winner = game['player1_username']
        winner_id = game['player1_id']
        loser_id = game['player2_id']
        player1_result = 'win'
        player2_result = 'lose'
    elif game['player2_score'] > game['player1_score']:
        final_winner = game['player2_username']
        winner_id = game['player2_id']
        loser_id = game['player1_id']
        player1_result = 'lose'
        player2_result = 'win'
    else:
        final_winner = None
        winner_id = None
        player1_result = 'draw'
        player2_result = 'draw'
    
    # Update user statistics
    player1 = User.query.get(game['player1_id'])
    player2 = User.query.get(game['player2_id'])
    
    if player1:
        player1.update_stats(player1_result)
    if player2:
        player2.update_stats(player2_result)
    
    # Update game session
    game_session = GameSession.query.filter_by(room_id=game_id).first()
    if game_session:
        game_session.winner_id = winner_id
        game_session.status = 'finished'
        game_session.ended_at = datetime.utcnow()
    
    db.session.commit()
    
    # Send game over event
    emit('game_over', {
        'winner': final_winner,
        'player1_score': game['player1_score'],
        'player2_score': game['player2_score'],
        'player1_username': game['player1_username'],
        'player2_username': game['player2_username']
    }, room=game_id)
    
    # Clean up game
    del active_games[game_id]

def handle_game_disconnect(game_id, disconnected_sid):
    if game_id not in active_games:
        return
    
    game = active_games[game_id]
    
    # Determine which player disconnected
    if disconnected_sid == game['player1_sid']:
        disconnected_player = game['player1_username']
        winner = game['player2_username']
    else:
        disconnected_player = game['player2_username']
        winner = game['player1_username']
    
    # Notify remaining player
    emit('player_disconnected', {
        'disconnected_player': disconnected_player,
        'winner': winner
    }, room=game_id)
    
    # Clean up game
    del active_games[game_id]

@app.route('/')
def home():
    return jsonify({
        'message': 'Rock Paper Scissors Game Server',
        'status': 'running',
        'timestamp': datetime.utcnow().isoformat()
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)