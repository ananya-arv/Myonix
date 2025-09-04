# real_time_myonix.py - Connect to your ESP32 and process data live
import bluetooth
import json
import time
import threading
from collections import deque
import numpy as np

class MyonixRealTimeProcessor:
    def __init__(self, buffer_size=300):  # 30 seconds at 10 Hz
        # Data buffers (circular buffers for real-time processing)
        self.eda_buffer = deque(maxlen=buffer_size)
        self.emg_buffer = deque(maxlen=buffer_size) 
        self.imu_buffer = deque(maxlen=buffer_size)
        self.timestamp_buffer = deque(maxlen=buffer_size)
        
        # Initialize your ML processor
        self.processor = MyonixMultiModalProcessor(sampling_rate=10)
        self.processor.load_model('myonix_model.joblib')  # Load pre-trained model
        
        # Bluetooth connection
        self.bt_socket = None
        self.is_connected = False
        
    def connect_to_esp32(self, mac_address="XX:XX:XX:XX:XX:XX"):
        """Connect to ESP32 via Bluetooth"""
        try:
            self.bt_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.bt_socket.connect((mac_address, 1))
            self.is_connected = True
            print("Connected to Myonix device!")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def collect_data_continuously(self):
        """Continuously collect data from ESP32"""
        while self.is_connected:
            try:
                # Read JSON data from ESP32
                data_string = self.bt_socket.recv(1024).decode('utf-8')
                
                # Parse multiple JSON objects (in case of buffered data)
                for line in data_string.strip().split('\n'):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            self.add_sensor_data(data)
                        except json.JSONDecodeError:
                            continue
                            
            except Exception as e:
                print(f"Data collection error: {e}")
                self.is_connected = False
    
    def add_sensor_data(self, data):
        """Add new sensor data to buffers"""
        self.eda_buffer.append(data['eda'])
        self.emg_buffer.append(data['emg'])
        
        imu_data = {
            'accel_x': data['accel_x'],
            'accel_y': data['accel_y'], 
            'accel_z': data['accel_z'],
            'gyro_x': data['gyro_x'],
            'gyro_y': data['gyro_y'],
            'gyro_z': data['gyro_z']
        }
        self.imu_buffer.append(imu_data)
        self.timestamp_buffer.append(data['timestamp'])
        
        # Process data every 30 samples (3 seconds)
        if len(self.eda_buffer) >= 30 and len(self.eda_buffer) % 30 == 0:
            self.process_current_window()
    
    def process_current_window(self):
        """Process current data window and get recommendations"""
        try:
            # Convert buffers to arrays
            eda_data = np.array(list(self.eda_buffer))
            emg_data = np.array(list(self.emg_buffer))
            
            # Convert IMU buffer to proper format
            imu_data = {
                'accel_x': [d['accel_x'] for d in self.imu_buffer],
                'accel_y': [d['accel_y'] for d in self.imu_buffer],
                'accel_z': [d['accel_z'] for d in self.imu_buffer],
                'gyro_x': [d['gyro_x'] for d in self.imu_buffer],
                'gyro_y': [d['gyro_y'] for d in self.imu_buffer],
                'gyro_z': [d['gyro_z'] for d in self.imu_buffer]
            }
            
            # Load into processor
            self.processor.load_sensor_data(eda_data=eda_data, 
                                          emg_data=emg_data, 
                                          imu_data=imu_data)
            
            # Get real-time recommendations
            recommendations = self.processor.get_real_time_recommendations()
            
            # Send recommendations to your game
            self.send_game_adaptations(recommendations)
            
            # Print status
            print(f"\n=== REAL-TIME ANALYSIS ===")
            print(f"State: {recommendations['predicted_state']}")
            print(f"Confidence: {recommendations['confidence']:.1%}")
            print(f"Action: {recommendations['recommendation']['action']}")
            print(f"Message: {recommendations['recommendation']['message']}")
            print(f"Stress: {recommendations['current_metrics']['stress_level']:.2f}")
            print(f"Effort: {recommendations['current_metrics']['effort_level']:.2f}")
            
        except Exception as e:
            print(f"Processing error: {e}")
    
    def send_game_adaptations(self, recommendations):
        """Send adaptation commands to your React game"""
        game_command = {
            'action': recommendations['recommendation']['action'],
            'parameters': recommendations['recommendation']['game_params'],
            'metrics': recommendations['current_metrics'],
            'message': recommendations['recommendation']['message']
        }
        
        # Send to your React app via WebSocket, HTTP API, or file
        # This is where you'd integrate with your Phase 4 game system
        self.send_to_react_game(game_command)
    
    def send_to_react_game(self, command):
        """Send command to React game (implement based on your architecture)"""
        # Option 1: WebSocket
        # websocket.send(json.dumps(command))
        
        # Option 2: HTTP POST to local React dev server
        # requests.post('http://localhost:3000/api/adaptation', json=command)
        
        # Option 3: Write to file that React polls
        with open('game_commands.json', 'w') as f:
            json.dump(command, f)
        
        print(f"Sent to game: {command['action']}")

# Usage example:
def start_myonix_system():
    processor = MyonixRealTimeProcessor()
    
    # Connect to your ESP32 (replace with your device's MAC address)
    if processor.connect_to_esp32("30:AE:A4:XX:XX:XX"):
        
        # Start data collection in separate thread
        collection_thread = threading.Thread(target=processor.collect_data_continuously)
        collection_thread.daemon = True
        collection_thread.start()
        
        print("Myonix system running! Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(1)  # Keep main thread alive
        except KeyboardInterrupt:
            processor.is_connected = False
            print("System stopped.")
    else:
        print("Failed to connect to Myonix device")

# Uncomment to run:
# start_myonix_system()