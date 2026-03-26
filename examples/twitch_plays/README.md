# Twitch Plays Go2

Control a Unitree Go2 quadruped via Twitch chat votes.

## Setup

1. **Get Twitch credentials** from [twitchtokengenerator.com](https://twitchtokengenerator.com/):
   - Select "Custom Scope Token"
   - Choose scopes: `chat:read`, `chat:edit`
   - Copy the Access Token

2. **Set environment variables**:
   ```bash
   export DIMOS_TWITCH_TOKEN=oauth:your_access_token_here
   export DIMOS_CHANNEL_NAME=your_twitch_channel
   ```

3. **Install twitchio** (not in DimOS base deps):
   ```bash
   uv pip install twitchio
   ```

## Run

```bash
# Real robot
dimos run unitree-go2-twitch --robot-ip 192.168.123.161

# Replay mode (no robot, test the chat integration)
dimos --replay run unitree-go2-twitch
```

## Chat Commands

Viewers type these in Twitch chat (with `!` prefix by default):

| Command | Action |
|---------|--------|
| `!forward` | Walk forward |
| `!back` | Walk backward |
| `!left` | Turn left |
| `!right` | Turn right |
| `!stop` | Stop moving |

## Voting Modes

Configure via `--vote-mode`:

| Mode | Behaviour |
|------|-----------|
| `plurality` (default) | Most votes wins. Classic "Twitch Plays" style. |
| `majority` | Winner needs >50% of votes. No action if split. |
| `weighted_recent` | Later votes in the window count more. Rewards fast reactions. |
| `runoff` | If no majority, top-2 enter instant runoff using each voter's latest vote. |

## Configuration

All configurable via CLI flags or env vars (prefixed `DIMOS_`):

| Flag | Default | Description |
|------|---------|-------------|
| `--vote-window-seconds` | 5.0 | Duration of each voting round |
| `--min-votes-threshold` | 1 | Minimum votes to trigger action |
| `--linear-speed` | 0.3 | Forward/backward speed (m/s) |
| `--angular-speed` | 0.5 | Turning speed (rad/s) |
| `--command-duration` | 1.0 | How long each command runs (s) |
| `--bot-prefix` | `!` | Chat command prefix |
| `--commands` | forward,back,left,right,stop | Valid keywords |

## Example: Local Testing Without Twitch

You can test the vote logic without a Twitch connection:

```python
from dimos.stream.twitch.module import TwitchChat

# Create module without credentials (local-only mode)
chat = TwitchChat(vote_window_seconds=2.0, vote_mode="weighted_recent")
chat.start()

# Simulate votes programmatically
chat.record_vote("forward", voter="user1")
chat.record_vote("forward", voter="user2")
chat.record_vote("left", voter="user3")

# The vote loop will tally and publish cmd_vel automatically
```
