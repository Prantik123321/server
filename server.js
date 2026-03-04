const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const mongoose = require('mongoose');
const cors = require('cors');
require('dotenv').config();

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: process.env.CORS_ORIGIN || "*",
    methods: ["GET", "POST"]
  },
  // Optimize for mobile/low bandwidth
  pingTimeout: 60000,
  pingInterval: 25000,
  transports: ['websocket', 'polling'] // Fallback support
});

// Middleware
app.use(cors());
app.use(express.json());

// Health check endpoint (Render needs this)
app.get('/', (req, res) => {
  res.json({ 
    status: 'online',
    players: gameState.players.size,
    uptime: process.uptime(),
    seed: gameState.worldSeed
  });
});

// MongoDB Connection with your credentials
mongoose.connect(process.env.MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
  serverSelectionTimeoutMS: 5000, // Timeout after 5s
  socketTimeoutMS: 45000, // Close sockets after 45s
}).then(() => {
  console.log('✅ Connected to MongoDB Atlas (Cluster0)');
  console.log('📊 Database: paperplane');
}).catch(err => {
  console.error('❌ MongoDB connection error:', err);
  console.log('⚠️ Server will continue without database - leaderboard disabled');
});

// Game state
const gameState = {
  players: new Map(), // Store connected players
  worldSeed: parseInt(process.env.WORLD_SEED) || Math.floor(Math.random() * 1000000),
  startTime: Date.now(),
  updateInterval: parseInt(process.env.UPDATE_INTERVAL) || 100
};

// Player Schema for MongoDB
const playerSchema = new mongoose.Schema({
  playerId: { type: String, required: true, unique: true },
  username: { type: String, required: true },
  highScore: { type: Number, default: 0 },
  totalPlayTime: { type: Number, default: 0 },
  gamesPlayed: { type: Number, default: 0 },
  lastSeen: { type: Date, default: Date.now },
  country: { type: String, default: 'Unknown' } // Optional
});

const Player = mongoose.model('Player', playerSchema);

// Leaderboard Schema (for caching)
const leaderboardSchema = new mongoose.Schema({
  updatedAt: { type: Date, default: Date.now },
  topPlayers: [{
    username: String,
    score: Number,
    playerId: String
  }]
});

const Leaderboard = mongoose.model('Leaderboard', leaderboardSchema);

// Cache leaderboard for 5 minutes
let leaderboardCache = {
  data: [],
  timestamp: 0,
  cacheTime: 5 * 60 * 1000 // 5 minutes
};

// Socket.io connection handling
io.on('connection', (socket) => {
  console.log(`🟢 Player connected: ${socket.id} | Total: ${gameState.players.size + 1}`);
  
  // Send world seed immediately on connection
  socket.emit('worldSeed', {
    seed: gameState.worldSeed,
    timestamp: Date.now()
  });

  // Send list of current players
  const currentPlayers = Array.from(gameState.players.values()).map(p => ({
    id: p.id,
    username: p.username,
    position: p.position,
    rotation: p.rotation,
    score: p.score || 0
  }));
  
  socket.emit('currentPlayers', currentPlayers);

  // Handle player join
  socket.on('playerJoin', async (data) => {
    try {
      const { username, country = 'Unknown' } = data;
      
      // Initialize player state
      const playerState = {
        id: socket.id,
        username: username || 'Player_' + socket.id.substr(0, 4),
        position: { x: 0, y: 5, z: 0 },
        rotation: { x: 0, y: 0, z: 0 },
        score: 0,
        lastUpdate: Date.now(),
        country: country
      };
      
      gameState.players.set(socket.id, playerState);
      
      // Try to save to database if connected
      if (mongoose.connection.readyState === 1) {
        try {
          await Player.findOneAndUpdate(
            { playerId: socket.id },
            { 
              username: playerState.username,
              lastSeen: Date.now(),
              country: country,
              $inc: { gamesPlayed: 1 }
            },
            { upsert: true, new: true }
          );
        } catch (dbError) {
          console.error('Database error on join:', dbError.message);
        }
      }
      
      // Broadcast new player to everyone except sender
      socket.broadcast.emit('playerJoined', {
        id: socket.id,
        username: playerState.username,
        position: playerState.position
      });
      
      console.log(`👤 Player registered: ${playerState.username} (${socket.id})`);
      
    } catch (error) {
      console.error('Error in playerJoin:', error);
      socket.emit('error', { message: 'Failed to join game' });
    }
  });

  // Handle position updates (with anti-cheat validation)
  socket.on('updatePosition', (data) => {
    const player = gameState.players.get(socket.id);
    if (!player) return;
    
    const now = Date.now();
    const timeDiff = now - player.lastUpdate;
    
    // Basic anti-cheat: validate movement speed
    if (timeDiff > 0) {
      const oldPos = player.position;
      const newPos = data.position;
      
      // Calculate distance moved
      const dx = newPos.x - oldPos.x;
      const dy = newPos.y - oldPos.y;
      const dz = newPos.z - oldPos.z;
      const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);
      
      // Max possible speed (adjust based on game physics)
      // 0.5 units per 100ms = 5 units per second
      const maxSpeed = 0.5; 
      
      if (distance > maxSpeed && timeDiff < 500) {
        console.warn(`⚠️ Anti-cheat: Player ${player.username} moved too fast: ${distance.toFixed(2)} units`);
        // Reject update or teleport back
        socket.emit('positionCorrection', player.position);
        return;
      }
      
      // Update player position if valid
      player.position = newPos;
      player.rotation = data.rotation;
      player.score = data.score || player.score || 0;
      player.lastUpdate = now;
      
      // Broadcast to other players
      socket.broadcast.emit('playerMoved', {
        id: socket.id,
        position: player.position,
        rotation: player.rotation,
        score: player.score
      });
    }
  });

  // Handle score updates (with validation)
  socket.on('updateScore', async (data) => {
    const player = gameState.players.get(socket.id);
    if (!player) return;
    
    const newScore = data.score;
    
    // Anti-cheat: Score can only increase and not too fast
    if (newScore > player.score) {
      const scoreDiff = newScore - player.score;
      const timeDiff = Date.now() - player.lastUpdate;
      
      // Max score gain per second (adjust based on game)
      const maxScorePerSecond = 10;
      const maxAllowedGain = (timeDiff / 1000) * maxScorePerSecond;
      
      if (scoreDiff <= maxAllowedGain + 1) { // +1 for rounding
        player.score = newScore;
        
        // Check if it's a new high score
        if (newScore > (player.highScore || 0)) {
          player.highScore = newScore;
          
          // Save to database
          if (mongoose.connection.readyState === 1) {
            try {
              await Player.findOneAndUpdate(
                { playerId: socket.id },
                { 
                  highScore: newScore,
                  lastSeen: Date.now()
                }
              );
              
              // Invalidate leaderboard cache
              leaderboardCache.timestamp = 0;
            } catch (dbError) {
              console.error('Error saving high score:', dbError.message);
            }
          }
        }
        
        // Broadcast score update
        io.emit('scoreUpdated', {
          id: socket.id,
          score: player.score
        });
      } else {
        console.warn(`⚠️ Anti-cheat: Player ${player.username} score gain too high: +${scoreDiff} in ${timeDiff}ms`);
        socket.emit('scoreCorrection', player.score);
      }
    }
  });

  // Handle leaderboard request
  socket.on('getLeaderboard', async (data) => {
    const limit = data?.limit || 10;
    
    // Check cache
    const now = Date.now();
    if (now - leaderboardCache.timestamp < leaderboardCache.cacheTime && 
        leaderboardCache.data.length >= limit) {
      socket.emit('leaderboardData', leaderboardCache.data.slice(0, limit));
      return;
    }
    
    // Fetch from database
    if (mongoose.connection.readyState === 1) {
      try {
        const topPlayers = await Player.find()
          .sort({ highScore: -1 })
          .limit(limit)
          .select('username highScore playerId -_id');
        
        leaderboardCache = {
          data: topPlayers,
          timestamp: now
        };
        
        socket.emit('leaderboardData', topPlayers);
      } catch (dbError) {
        console.error('Error fetching leaderboard:', dbError);
        socket.emit('leaderboardData', []);
      }
    } else {
      // Return in-memory scores if DB unavailable
      const topPlayers = Array.from(gameState.players.values())
        .sort((a, b) => (b.score || 0) - (a.score || 0))
        .slice(0, limit)
        .map(p => ({
          username: p.username,
          highScore: p.score || 0,
          playerId: p.id
        }));
      
      socket.emit('leaderboardData', topPlayers);
    }
  });

  // Handle disconnection
  socket.on('disconnect', async () => {
    const player = gameState.players.get(socket.id);
    
    if (player) {
      console.log(`🔴 Player disconnected: ${player.username} (${socket.id})`);
      
      // Update play time in database
      if (mongoose.connection.readyState === 1) {
        try {
          const playTime = Math.floor((Date.now() - player.lastUpdate) / 1000);
          await Player.findOneAndUpdate(
            { playerId: socket.id },
            { 
              lastSeen: Date.now(),
              $inc: { totalPlayTime: playTime }
            }
          );
        } catch (dbError) {
          console.error('Error updating play time:', dbError.message);
        }
      }
      
      gameState.players.delete(socket.id);
      
      // Notify others
      io.emit('playerLeft', socket.id);
    }
  });
});

// Periodic cleanup and stats (every 5 minutes)
setInterval(() => {
  const now = Date.now();
  const timeout = 30000; // 30 seconds
  
  // Remove stale players
  for (const [id, player] of gameState.players.entries()) {
    if (now - player.lastUpdate > timeout) {
      console.log(`🧹 Removing stale player: ${player.username}`);
      gameState.players.delete(id);
      io.emit('playerLeft', id);
    }
  }
  
  console.log(`📊 Server stats: ${gameState.players.size} active players`);
}, 300000);

// Handle Render's sleep mode with keep-alive
const PORT = process.env.PORT || 10000;
server.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`🎲 World seed: ${gameState.worldSeed}`);
  console.log(`💾 MongoDB: ${mongoose.connection.readyState === 1 ? 'Connected' : 'Disconnected'}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, closing connections...');
  server.close(() => {
    mongoose.disconnect();
    console.log('Server shutdown complete');
    process.exit(0);
  });
});