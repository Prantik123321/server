const express = require("express");
const http = require("http");
const WebSocket = require("ws");
const cors = require("cors");

const app = express();
app.use(cors());

const PORT = process.env.PORT || 3000;

const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

let players = {};

app.get("/", (req, res) => {
  res.send("WebSocket Multiplayer Server Running");
});

wss.on("connection", (ws) => {
  const id = Math.random().toString(36).substr(2, 9);

  players[id] = { id, x: 0, y: 0, z: 0, ry: 0 };

  console.log("Player connected:", id);

  // Send current players
  ws.send(JSON.stringify({
    type: "init",
    id: id,
    players: players
  }));

  ws.on("message", (message) => {
    const data = JSON.parse(message);

    if (data.type === "update") {
      players[id] = {
        id: id,
        x: data.x,
        y: data.y,
        z: data.z,
        ry: data.ry
      };

      broadcast({
        type: "move",
        player: players[id]
      });
    }
  });

  ws.on("close", () => {
    console.log("Player disconnected:", id);
    delete players[id];

    broadcast({
      type: "disconnect",
      id: id
    });
  });
});

function broadcast(data) {
  const msg = JSON.stringify(data);
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(msg);
    }
  });
}

server.listen(PORT, () => {
  console.log("Server running on port", PORT);
});
