const mongoose = require('mongoose');

// Optional: Add database helper functions
class DatabaseHelper {
  constructor() {
    this.isConnected = false;
  }

  async getPlayerStats(playerId) {
    try {
      const Player = mongoose.model('Player');
      const stats = await Player.findOne({ playerId });
      return stats;
    } catch (error) {
      console.error('Error getting player stats:', error);
      return null;
    }
  }

  async getGlobalStats() {
    try {
      const Player = mongoose.model('Player');
      const stats = await Player.aggregate([
        {
          $group: {
            _id: null,
            totalPlayers: { $sum: 1 },
            averageScore: { $avg: '$highScore' },
            totalPlayTime: { $sum: '$totalPlayTime' }
          }
        }
      ]);
      return stats[0] || { totalPlayers: 0, averageScore: 0, totalPlayTime: 0 };
    } catch (error) {
      console.error('Error getting global stats:', error);
      return null;
    }
  }
}

module.exports = new DatabaseHelper();