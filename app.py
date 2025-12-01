from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import os
from datetime import datetime
import random
import json
import sqlite3
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-12345-change-this')

CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize SocketIO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25
)

# SQLite Database setup
DB_FILE = 'rps_game.db'

# ========== DATABASE HELPER FUNCTIONS ==========
def init_db():
    """Initialize SQLite database"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                draws INTEGER DEFAULT 0,
                total_games INTEGER DEFAULT 0,
                score INTEGER DEFAULT 1000,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                last_played TIMESTAMP
            )
        ''')
        
        # Create game_sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT UNIQUE NOT NULL,
                player1_id INTEGER,
                player2_id INTEGER,
                player1_username TEXT,
                player2_username TEXT,
                player1_move TEXT,
                player2_move TEXT,
                winner_id INTEGER,
                result TEXT,
                rounds INTEGER DEFAULT 1,
                player1_score INTEGER DEFAULT 0,
                player2_score INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP
            )
        ''')
        
        # Create leaderboard table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS leaderboard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE NOT NULL,
                username TEXT NOT NULL,
                score INTEGER DEFAULT 1000,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_games INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

# ========== DATABASE OPERATIONS ==========
def create_user(username, password):
    """Create new user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if username exists
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            conn.close()
            return None, "Username already exists"
        
        # Create user
        password_hash = hash_password(password)
        cursor.execute('''
            INSERT INTO users (username, password_hash, created_at)
            VALUES (?, ?, ?)
        ''', (username, password_hash, datetime.utcnow()))
        
        user_id = cursor.lastrowid
        
        # Create leaderboard entry
        cursor.execute('''
            INSERT INTO leaderboard (user_id, username, score)
            VALUES (?, ?, ?)
        ''', (user_id, username, 1000))
        
        conn.commit()
        conn.close()
        
        return user_id, None
    except Exception as e:
        logger.error(f"Create user error: {e}")
        return None, "Internal server error"

def authenticate_user(username, password):
    """Authenticate user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, password_hash, wins, losses, draws, score
            FROM users WHERE username = ?
        ''', (username,))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            if hash_password(password) == user['password_hash']:
                # Update last login
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                             (datetime.utcnow(), user['id']))
                conn.commit()
                conn.close()
                
                return {
                    'id': user['id'],
                    'username': user['username'],
                    'wins': user['wins'],
                    'losses': user['losses'],
                    'draws': user['draws'],
                    'score': user['score'],
                    'total_games': user['wins'] + user['losses'] + user['draws']
                }, None
        return None, "Invalid username or password"
    except Exception as e:
        logger.error(f"Authenticate user error: {e}")
        return None, "Internal server error"

def get_user_stats(user_id):
    """Get user statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT username, wins, losses, draws, score, created_at
            FROM users WHERE id = ?
        ''', (user_id,))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            total_games = user['wins'] + user['losses'] + user['draws']
            win_rate = round((user['wins'] / total_games * 100), 2) if total_games > 0 else 0
            
            return {
                'user_id': user_id,
                'username': user['username'],
                'wins': user['wins'],
                'losses': user['losses'],
                'draws': user['draws'],
                'total_games': total_games,
                'win_rate': win_rate,
                'score': user['score'],
                'created_at': user['created_at']
            }
        return None
    except Exception as e:
        logger.error(f"Get user stats error: {e}")
        return None

def update_user_stats(user_id, result, score_change=0):
    """Update user statistics after game"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if result == 'win':
            cursor.execute('''
                UPDATE users SET wins = wins + 1, total_games = total_games + 1,
                score = score + ?, last_played = ?
                WHERE id = ?
            ''', (score_change, datetime.utcnow(), user_id))
        elif result == 'loss':
            cursor.execute('''
                UPDATE users SET losses = losses + 1, total_games = total_games + 1,
                score = score - ?, last_played = ?
                WHERE id = ?
            ''', (abs(score_change), datetime.utcnow(), user_id))
        elif result == 'draw':
            cursor.execute('''
                UPDATE users SET draws = draws + 1, total_games = total_games + 1,
                last_played = ?
                WHERE id = ?
            ''', (datetime.utcnow(), user_id))
        
        # Update leaderboard
        cursor.execute('''
            UPDATE leaderboard SET
            wins = (SELECT wins FROM users WHERE id = ?),
            losses = (SELECT losses FROM users WHERE id = ?),
            total_games = (SELECT total_games FROM users WHERE id = ?),
            score = (SELECT score FROM users WHERE id = ?),
            win_rate = ROUND((SELECT wins FROM users WHERE id = ?) * 100.0 / 
                     NULLIF((SELECT total_games FROM users WHERE id = ?), 0), 2),
            last_updated = ?
            WHERE user_id = ?
        ''', (user_id, user_id, user_id, user_id, user_id, user_id, datetime.utcnow(), user_id))
        
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Update user stats error: {e}")

def get_leaderboard(limit=10):
    """Get leaderboard"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, username, score, wins, losses, total_games, win_rate
            FROM leaderboard
            ORDER BY score DESC
            LIMIT ?
        ''', (limit,))
        
        leaders = cursor.fetchall()
        conn.close()
        
        return [
            {
                'rank': idx + 1,
                'user_id': row['user_id'],
                'username': row['username'],
                'score': row['score'],
                'wins': row['wins'],
                'losses': row['losses'],
                'total_games': row['total_games'],
                'win_rate': row['win_rate'] or 0
            }
            for idx, row in enumerate(leaders)
        ]
    except Exception as e:
        logger.error(f"Get leaderboard error: {e}")
        return []

def save_game_session(game_data):
    """Save game session to database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO game_sessions (
                game_id, player1_id, player2_id, player1_username, player2_username,
                player1_move, player2_move, winner_id, result, rounds,
                player1_score, player2_score, created_at, ended_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            game_data['game_id'],
            game_data.get('player1_id'),
            game_data.get('player2_id'),
            game_data.get('player1_username'),
            game_data.get('player2_username'),
            game_data.get('player1_move'),
            game_data.get('player2_move'),
            game_data.get('winner_id'),
            game_data.get('result'),
            game_data.get('rounds', 1),
            game_data.get('player1_score', 0),
            game_data.get('player2_score', 0),
            game_data.get('created_at'),
            datetime.utcnow()
        ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Save game session error: {e}")

def get_server_stats():
    """Get server statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) as total_users FROM users')
        total_users = cursor.fetchone()['total_users']
        
        cursor.execute('SELECT COUNT(*) as total_games FROM game_sessions')
        total_games = cursor.fetchone()['total_games']
        
        conn.close()
        
        return {
            'total_users': total_users,
            'total_games': total_games,
            'active_players': len(user_socket_map),
            'waiting_players': len(waiting_players),
            'active_games': len(active_games)
        }
    except Exception as e:
        logger.error(f"Get server stats error: {e}")
        return {}

# ========== GAME STATE MANAGEMENT ==========
active_games = {}
waiting_players = {}
user_socket_map = {}

# ========== FLASK ROUTES ==========
@app.route('/')
def home():
    return jsonify({
        "message": "Rock Paper Scissors Server",
        "status": "online",
        "version": "1.0.0",
        "docs": {
            "register": "POST /api/register",
            "login": "POST /api/login",
            "profile": "GET /api/profile/<user_id>",
            "leaderboard": "GET /api/leaderboard",
            "stats": "GET /api/stats",
            "health": "GET /api/health"
        }
    })

@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "server": "Rock Paper Scissors"
    })

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if len(username) < 3:
            return jsonify({"error": "Username must be at least 3 characters"}), 400
        
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400
        
        user_id, error = create_user(username, password)
        if error:
            return jsonify({"error": error}), 400
        
        return jsonify({
            "message": "Registration successful",
            "user_id": user_id,
            "username": username
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        user_data, error = authenticate_user(username, password)
        if error:
            return jsonify({"error": error}), 401
        
        return jsonify({
            "message": "Login successful",
            "user": user_data
        })
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/profile/<int:user_id>')
def profile(user_id):
    try:
        user_stats = get_user_stats(user_id)
        if not user_stats:
            return jsonify({"error": "User not found"}), 404
        
        return jsonify(user_stats)
    except Exception as e:
        logger.error(f"Profile error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/leaderboard')
def leaderboard():
    try:
        leaders = get_leaderboard(10)
        return jsonify(leaders)
    except Exception as e:
        logger.error(f"Leaderboard error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/stats')
def stats():
    try:
        server_stats = get_server_stats()
        return jsonify(server_stats)
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ========== SOCKET.IO EVENTS ==========
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {
        'message': 'Connected to Rock Paper Scissors server',
        'sid': request.sid
    })

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")
    
    # Remove from waiting players
    if request.sid in waiting_players:
        del waiting_players[request.sid]
    
    # Remove from user-socket map
    for user_id, sid in list(user_socket_map.items()):
        if sid == request.sid:
            del user_socket_map[user_id]
            break
    
    # Handle game disconnection
    for game_id in list(active_games.keys()):
        game = active_games[game_id]
        if request.sid == game['player1_sid']:
            opponent_sid = game['player2_sid']
            emit('opponent_disconnected', {'message': 'Opponent disconnected'}, room=opponent_sid)
            cleanup_game(game_id)
            break
        elif request.sid == game['player2_sid']:
            opponent_sid = game['player1_sid']
            emit('opponent_disconnected', {'message': 'Opponent disconnected'}, room=opponent_sid)
            cleanup_game(game_id)
            break

@socketio.on('authenticate')
def handle_authenticate(data):
    user_id = data.get('user_id')
    username = data.get('username')
    
    if user_id and username:
        user_socket_map[user_id] = request.sid
        emit('authenticated', {
            'message': 'Authentication successful',
            'user_id': user_id,
            'username': username
        })
        logger.info(f"User authenticated: {username}")

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
        emit('status', {'message': 'Already searching for match'})
        return
    
    # Add to waiting list
    waiting_players[request.sid] = {
        'user_id': user_id,
        'username': username,
        'sid': request.sid
    }
    
    emit('searching', {
        'message': 'Looking for opponent...',
        'players_in_queue': len(waiting_players)
    })
    
    logger.info(f"Player {username} started matchmaking")
    
    # Try to match immediately
    if len(waiting_players) >= 2:
        create_match()

@socketio.on('cancel_matchmaking')
def handle_cancel_matchmaking():
    if request.sid in waiting_players:
        player_data = waiting_players.pop(request.sid)
        emit('matchmaking_cancelled', {
            'message': 'Matchmaking cancelled',
            'username': player_data['username']
        })
        logger.info(f"Player {player_data['username']} cancelled matchmaking")

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
    valid_choices = ['rock', 'paper', 'scissors']
    
    if choice not in valid_choices:
        emit('error', {'message': 'Invalid choice'})
        return
    
    # Record choice
    if request.sid == game['player1_sid']:
        game['player1_choice'] = choice
    elif request.sid == game['player2_sid']:
        game['player2_choice'] = choice
    
    emit('choice_made', {'choice': choice})
    
    # Notify opponent
    opponent_sid = game['player2_sid'] if request.sid == game['player1_sid'] else game['player1_sid']
    emit('opponent_choosing', {}, room=opponent_sid)
    
    # Check if both made choices
    if game.get('player1_choice') and game.get('player2_choice'):
        determine_winner(game_id)

@socketio.on('play_again')
def handle_play_again(data):
    game_id = data.get('game_id')
    
    if game_id in active_games:
        game = active_games[game_id]
        game['player1_choice'] = None
        game['player2_choice'] = None
        game['round'] += 1
        
        emit('next_round', {
            'round': game['round'],
            'scores': {
                'player1': game['player1_score'],
                'player2': game['player2_score']
            }
        }, room=game_id)

@socketio.on('leave_game')
def handle_leave_game(data):
    game_id = data.get('game_id')
    
    if game_id in active_games:
        cleanup_game(game_id)
        emit('game_left', {'message': 'You left the game'})

@socketio.on('send_message')
def handle_chat(data):
    game_id = data.get('game_id')
    message = data.get('message')
    username = data.get('username')
    
    if game_id and message and username:
        emit('receive_message', {
            'username': username,
            'message': message,
            'timestamp': datetime.utcnow().isoformat()
        }, room=game_id, skip_sid=request.sid)

# ========== GAME LOGIC FUNCTIONS ==========
def create_match():
    """Create a match between two waiting players"""
    if len(waiting_players) < 2:
        return
    
    waiting_list = list(waiting_players.items())
    player1_sid, player1_data = waiting_list[0]
    player2_sid, player2_data = waiting_list[1]
    
    # Remove from waiting
    waiting_players.pop(player1_sid)
    waiting_players.pop(player2_sid)
    
    # Create game ID
    game_id = f"game_{int(datetime.now().timestamp())}_{random.randint(1000, 9999)}"
    
    # Create game
    active_games[game_id] = {
        'player1_sid': player1_sid,
        'player2_sid': player2_sid,
        'player1_id': player1_data['user_id'],
        'player2_id': player2_data['user_id'],
        'player1_username': player1_data['username'],
        'player2_username': player2_data['username'],
        'player1_choice': None,
        'player2_choice': None,
        'player1_score': 0,
        'player2_score': 0,
        'round': 1,
        'created_at': datetime.utcnow().isoformat()
    }
    
    # Join rooms
    join_room(game_id, player1_sid)
    join_room(game_id, player2_sid)
    
    # Notify players
    emit('match_found', {
        'game_id': game_id,
        'opponent': player2_data['username'],
        'message': 'Match found! Game starting...'
    }, room=player1_sid)
    
    emit('match_found', {
        'game_id': game_id,
        'opponent': player1_data['username'],
        'message': 'Match found! Game starting...'
    }, room=player2_sid)
    
    # Start game
    emit('game_start', {
        'game_id': game_id,
        'opponent': player2_data['username'],
        'round': 1
    }, room=player1_sid)
    
    emit('game_start', {
        'game_id': game_id,
        'opponent': player1_data['username'],
        'round': 1
    }, room=player2_sid)
    
    logger.info(f"Match created: {player1_data['username']} vs {player2_data['username']}")

def determine_winner(game_id):
    """Determine winner of a game round"""
    game = active_games[game_id]
    p1_choice = game['player1_choice']
    p2_choice = game['player2_choice']
    
    # Determine result
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
    
    # Update database stats
    if result == 'player1_win':
        update_user_stats(game['player1_id'], 'win', 10)
        update_user_stats(game['player2_id'], 'loss', 5)
    elif result == 'player2_win':
        update_user_stats(game['player2_id'], 'win', 10)
        update_user_stats(game['player1_id'], 'loss', 5)
    else:
        update_user_stats(game['player1_id'], 'draw', 0)
        update_user_stats(game['player2_id'], 'draw', 0)
    
    # Prepare result data
    result_data = {
        'player1': {
            'username': game['player1_username'],
            'choice': p1_choice,
            'score': game['player1_score']
        },
        'player2': {
            'username': game['player2_username'],
            'choice': p2_choice,
            'score': game['player2_score']
        },
        'result': result,
        'winner': winner,
        'round': game['round'],
        'game_id': game_id
    }
    
    # Send result
    emit('round_result', result_data, room=game_id)
    
    # Save game session
    save_game_session({
        'game_id': game_id,
        'player1_id': game['player1_id'],
        'player2_id': game['player2_id'],
        'player1_username': game['player1_username'],
        'player2_username': game['player2_username'],
        'player1_move': p1_choice,
        'player2_move': p2_choice,
        'winner_id': game['player1_id'] if result == 'player1_win' else game['player2_id'] if result == 'player2_win' else None,
        'result': result,
        'rounds': game['round'],
        'player1_score': game['player1_score'],
        'player2_score': game['player2_score'],
        'created_at': game['created_at']
    })
    
    # Check if game over (best of 3)
    if game['player1_score'] >= 2 or game['player2_score'] >= 2:
        final_winner = game['player1_username'] if game['player1_score'] >= 2 else game['player2_username']
        emit('game_over', {
            'winner': final_winner,
            'final_scores': {
                game['player1_username']: game['player1_score'],
                game['player2_username']: game['player2_score']
            },
            'game_id': game_id
        }, room=game_id)
        
        # Clean up after delay
        socketio.start_background_task(cleanup_game_delayed, game_id)
    else:
        # Reset choices for next round
        game['player1_choice'] = None
        game['player2_choice'] = None
        game['round'] += 1

def cleanup_game(game_id):
    """Clean up game immediately"""
    if game_id in active_games:
        game = active_games[game_id]
        leave_room(game_id, game['player1_sid'])
        leave_room(game_id, game['player2_sid'])
        del active_games[game_id]
        logger.info(f"Game {game_id} cleaned up")

def cleanup_game_delayed(game_id):
    """Clean up game after delay"""
    import time
    time.sleep(5)
    cleanup_game(game_id)

# ========== INITIALIZATION ==========
@app.before_first_request
def initialize():
    """Initialize database on first request"""
    init_db()
    logger.info("Server initialized")

# ========== MAIN ==========
if __name__ == '__main__':
    # Initialize database
    init_db()
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    
    socketio.run(
        app,
        host='0.0.0.0',
        port=port,
        debug=False,
        log_output=True
    )