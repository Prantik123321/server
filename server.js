const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const cors = require("cors");

const app = express();
app.use(cors());

// Render requires dynamic port
const PORT = process.env.PORT || 3000;

const server = http.createServer(app);

const io = new Server(server, {
  cors: {
    origin: "*",
  },
});

// In-memory player store
let players = {};

// Basic health check
app.get("/", (req, res) => {
  res.send("Multiplayer server is running");
});

io.on("connection", (socket) => {
  console.log("Player connected:", socket.id);

  // Create new player
  players[socket.id] = {
    id: socket.id,
    x: 0,
    y: 0,
    z: 0,
    ry: 0
  };

  // Send existing players to new player
  socket.emit("currentPlayers", players);

  // Notify others
  socket.broadcast.emit("newPlayer", players[socket.id]);

  // Receive position update
  socket.on("updatePosition", (data) => {
    if (!players[socket.id]) return;

    // Basic anti-cheat (ignore teleport jump)
    const old = players[socket.id];

    const dx = Math.abs(data.x - old.x);
    const dy = Math.abs(data.y - old.y);
    const dz = Math.abs(data.z - old.z);

    if (dx > 20 || dy > 20 || dz > 20) {
      console.log("Teleport detected, ignoring");
      return;
    }

    players[socket.id] = {
      id: socket.id,
      x: data.x,
      y: data.y,
      z: data.z,
      ry: data.ry
    };

    // Broadcast to others
    socket.broadcast.emit("playerMoved", players[socket.id]);
  });

  socket.on("disconnect", () => {
    console.log("Player disconnected:", socket.id);
    delete players[socket.id];
    io.emit("playerDisconnected", socket.id);
  });
});

server.listen(PORT, () => {
  console.log("Server running on port", PORT);
});
