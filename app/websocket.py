from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
from datetime import datetime

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, List[WebSocket]] = {}
        self.user_connections: Dict[int, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
        self.user_connections[user_id] = websocket

    def disconnect(self, websocket: WebSocket, user_id: int):
        if user_id in self.active_connections:
            self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        if user_id in self.user_connections:
            del self.user_connections[user_id]

    async def send_personal_message(self, message: dict, user_id: int):
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                await connection.send_json(message)

    async def broadcast(self, message: dict, exclude_user_id: int = None):
        for user_id, connections in self.active_connections.items():
            if user_id != exclude_user_id:
                for connection in connections:
                    await connection.send_json(message)

manager = ConnectionManager()

async def websocket_endpoint(websocket: WebSocket, user_id: int):
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle different message types
            if message_data.get("type") == "typing":
                # Notify other users in chat
                pass
            elif message_data.get("type") == "message":
                # Broadcast message
                await manager.broadcast(message_data, exclude_user_id=user_id)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)