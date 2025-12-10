#!/usr/bin/env python3
"""
Event Execution Logger

Subscribes to MQTT topic `beat/events/execution_log` and writes event execution
data to CSV files for analysis. Logs include scheduled time, actual execution
time, event type, and whether events are automatic turn-offs.

Usage:
    python event_execution_logger.py [--broker BROKER] [--port PORT] [--output-dir DIR]
"""

import paho.mqtt.client as mqtt
import json
import csv
import os
import argparse
import sys
import signal
from datetime import datetime
from typing import Optional


class EventExecutionLogger:
    """MQTT subscriber that logs event execution data to CSV"""
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883,
                 output_dir: str = "logs"):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.output_dir = output_dir
        self.client = None
        self.connected = False
        self.csv_file = None
        self.csv_writer = None
        self.log_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(self.output_dir, f"event_execution_{timestamp}.csv")
        
    def connect(self) -> bool:
        """Connect to MQTT broker"""
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            
            print(f"Connecting to MQTT broker at {self.broker_host}:{self.broker_port}...")
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            
            # Wait for connection
            import time
            timeout = 5
            start = time.time()
            while not self.connected and (time.time() - start) < timeout:
                time.sleep(0.1)
            
            if self.connected:
                print("Connected to MQTT broker")
                # Subscribe to execution log topic
                self.client.subscribe("beat/events/execution_log", qos=0)
                print("Subscribed to topic: beat/events/execution_log")
                return True
            else:
                print("Failed to connect to MQTT broker")
                return False
                
        except Exception as e:
            print(f"Error connecting to MQTT broker: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker and close CSV file"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            print("Disconnected from MQTT broker")
        
        if self.csv_file:
            self.csv_file.close()
            print(f"CSV file closed: {self.filename}")
            print(f"Total events logged: {self.log_count}")
    
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        """MQTT connection callback"""
        if reason_code == 0:
            self.connected = True
            print("MQTT connection successful")
        else:
            print(f"MQTT connection failed with code {reason_code}")
            self.connected = False
    
    def _on_disconnect(self, client, userdata, reason_code, properties=None, *args, **kwargs):
        """MQTT disconnection callback"""
        self.connected = False
        if reason_code != 0:
            print(f"Unexpected MQTT disconnection (code: {reason_code})")
    
    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages"""
        try:
            # Parse JSON message
            data = json.loads(msg.payload.decode('utf-8'))
            
            # Extract fields
            event_id = data.get("event_id", "")
            event_type = data.get("event_type", "")
            is_automatic = data.get("is_automatic", False)
            scheduled_time = data.get("scheduled_time", "0.000000")
            actual_time = data.get("actual_time", "0.000000")
            
            # Open CSV file if not already open
            if self.csv_file is None:
                self.csv_file = open(self.filename, 'w', newline='')
                self.csv_writer = csv.writer(self.csv_file)
                # Write header
                self.csv_writer.writerow([
                    "event_id",
                    "event_type",
                    "is_automatic",
                    "scheduled_time",
                    "actual_time"
                ])
                print(f"CSV file opened: {self.filename}")
            
            # Write data row
            self.csv_writer.writerow([
                event_id,
                event_type,
                is_automatic,
                scheduled_time,
                actual_time
            ])
            
            # Flush to ensure data is written immediately
            self.csv_file.flush()
            
            self.log_count += 1
            if self.log_count % 10 == 0:
                print(f"Logged {self.log_count} events...")
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON message: {e}")
            print(f"Message: {msg.payload.decode('utf-8')}")
        except Exception as e:
            print(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Subscribe to MQTT execution logs and write to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default broker (localhost:1883)
  python event_execution_logger.py
  
  # Custom broker and output directory
  python event_execution_logger.py --broker 192.168.1.100 --port 1883 --output-dir logs/
        """
    )
    parser.add_argument(
        "--broker",
        default="localhost",
        help="MQTT broker hostname (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1883,
        help="MQTT broker port (default: 1883)"
    )
    parser.add_argument(
        "--output-dir",
        default="logs",
        help="Directory for CSV log files (default: logs)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Event Execution Logger")
    print("=" * 60)
    print(f"Broker: {args.broker}:{args.port}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Create logger
    logger = EventExecutionLogger(
        broker_host=args.broker,
        broker_port=args.port,
        output_dir=args.output_dir
    )
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down...")
        logger.disconnect()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Connect to broker
    if not logger.connect():
        print("ERROR: Failed to connect to MQTT broker")
        print("Make sure Mosquitto is running:")
        print("  mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf")
        sys.exit(1)
    
    try:
        print("\nListening for event execution logs...")
        print("Press Ctrl+C to stop\n")
        
        # Keep running
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        logger.disconnect()


if __name__ == "__main__":
    main()

