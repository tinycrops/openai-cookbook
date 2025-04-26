#!/usr/bin/env python3
"""
Demo for SignalRecorder: records and fetches example signals.
"""
from signal_recorder import SignalRecorder

def main():
    recorder = SignalRecorder()
    # Record some example signals
    recorder.record_signal('channel_capacity', 100.0, {'unit': 'bits/s', 'source': 'Shannon'})
    recorder.record_signal('episode_reward', 42.5, {'unit': 'score', 'description': 'RL episode reward'})
    # Fetch and display recent signals
    signals = recorder.fetch_signals(limit=10)
    for sig in signals:
        print(f"[{sig['timestamp']}] {sig['signal_name']} = {sig['signal_value']} metadata={sig['metadata']}")
    recorder.close()

if __name__ == '__main__':
    main()