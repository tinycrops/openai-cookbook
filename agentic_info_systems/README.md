# Agentic Information Systems

This module provides tools to record important signals and metrics for agentic information systems.

## Components

- **signal_recorder.py**: Defines `SignalRecorder`, a simple SQLite-based recorder for signals.
- **demo.py**: Demonstrates recording and fetching signals.

## Usage

1. Navigate to the `agentic_info_systems` directory:
   ```bash
   cd agentic_info_systems
   ```
2. Run the demo:
   ```bash
   python demo.py
   ```
3. Integrate `SignalRecorder` into your agentic system to record signals:
   ```python
   from signal_recorder import SignalRecorder

   recorder = SignalRecorder(db_path='path/to/signals.db')
   recorder.record_signal('entropy', 3.14, {'unit': 'bits'})
   ...
   recorder.close()
   ```

Signals are stored in `signals.db` (default) with the schema:

```sql
CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    signal_name TEXT NOT NULL,
    signal_value REAL,
    metadata TEXT
);
```