# Enhanced Agentic Insights Implementation

This project extends the original `agentic_insights.py` with advanced reinforcement learning techniques and real-world actuators based on PhD committee feedback.

## Key Improvements

1. **Neural Network World Model**
   - Replaced simple table-based lookups with an embedding-based neural model
   - Generalizes better across states and actions
   - Provides more accurate predictions for novel situations

2. **Q-Learning & DQN for Action Selection**
   - Implemented Deep Q-Network for optimal action selection
   - Treats Gemini's candidate actions as a policy prior
   - Uses Q-learning to fine-tune selection with expected future rewards

3. **Real-World Actuation through Spotify**
   - The agent can now control music via Spotify API
   - Plays focus/productivity playlists when appropriate
   - Monitors user behavior (e.g., skipping tracks) as additional reward signal

4. **Experiment Tracking & Performance Visualization**
   - Added comprehensive tracking of experiments
   - Visualization tools for comparing different agent configurations
   - Ablation study tool to quantify the impact of each component

## Running Ablation Studies

To compare the performance of different agent configurations:

```bash
python scripts/ablation_study.py
```

This will run a series of experiments comparing:
- Baseline (table model with greedy selection)
- Neural world model (without DQN)
- DQN action selection (without neural model)
- Full system (neural model + DQN + Spotify)

Results will be displayed as graphs showing cumulative rewards, moving averages, and action distributions.

## Configuration

The following environment variables can be set:
- `GEMINI_API_KEY`: API key for Google Gemini models
- `SPOTIFY_CLIENT_ID`: Spotify developer API client ID
- `SPOTIFY_CLIENT_SECRET`: Spotify developer API client secret
- `SPOTIFY_REDIRECT_URI`: OAuth redirect URI for Spotify authentication

## Dependencies

Additional dependencies beyond the original project:
- PyTorch: For neural network implementation
- Matplotlib: For visualization
- Spotipy: For Spotify API integration

Install dependencies with:
```bash
pip install torch matplotlib spotipy
```

## Next Steps

Future improvements could include:
1. Longer planning horizon using Monte Carlo Tree Search
2. Additional real-world actuators (e.g., lighting, calendars)
3. More diverse reward signals from physical sensors
4. Meta-learning to adapt reward weights automatically 