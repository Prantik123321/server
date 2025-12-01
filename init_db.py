import sqlite3
import hashlib
from datetime import datetime

def init_database():
    """Initialize the database"""
    DB_FILE = 'rps_game.db'
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create tables
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
    
    # Create a test user
    def hash_password(password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    # Check if test user exists
    cursor.execute("SELECT id FROM users WHERE username = 'test'")
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            ('test', hash_password('test123'))
        )
        
        user_id = cursor.lastrowid
        cursor.execute(
            "INSERT INTO leaderboard (user_id, username) VALUES (?, ?)",
            (user_id, 'test')
        )
        
        print(f"✅ Created test user: username='test', password='test123'")
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully!")

if __name__ == "__main__":
    init_database()