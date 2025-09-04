# myonix_system.py - Complete integration script
import time
import threading
from myonix_processor import MyonixMultiModalProcessor
from real_time_processor import MyonixRealTimeProcessor
import json
from flask import Flask, jsonify
from flask_cors import CORS

# Create Flask app for React communication
app = Flask(__name__)
CORS(app)

# Global variables
current_recommendations = {}
system_status = {"running": False, "connected": False}

@app.route('/api/game-commands')
def get_game_commands():
    """API endpoint for React to get current recommendations"""
    return jsonify(current_recommendations)

@app.route('/api/system-status')
def get_system_status():
    """Get system status"""
    return jsonify(system_status)

@app.route('/api/patient-metrics')
def get_patient_metrics():
    """Get current patient metrics"""
    if 'current_metrics' in current_recommendations:
        return jsonify(current_recommendations['current_metrics'])
    return jsonify({"error": "No metrics available"})

class MyonixSystemManager:
    def __init__(self):
        self.processor = MyonixRealTimeProcessor()
        self.flask_thread = None
        
    def start_complete_system(self, esp32_mac="30:AE:A4:XX:XX:XX"):
        """Start the complete Myonix system"""
        global system_status, current_recommendations
        
        print("Starting Myonix System...")
        
        # 1. Connect to ESP32
        if self.processor.connect_to_esp32(esp32_mac):
            system_status["connected"] = True
            print("✓ Connected to ESP32")
            
            # 2. Start data collection
            collection_thread = threading.Thread(
                target=self.processor.collect_data_continuously
            )
            collection_thread.daemon = True
            collection_thread.start()
            print("✓ Data collection started")
            
            # 3. Start Flask API for React communication
            self.flask_thread = threading.Thread(
                target=lambda: app.run(host='0.0.0.0', port=5000, debug=False)
            )
            self.flask_thread.daemon = True
            self.flask_thread.start()
            print("✓ API server started on http://localhost:5000")
            
            # 4. Override the send_game_adaptations method to update global state
            original_send = self.processor.send_game_adaptations
            def enhanced_send(recommendations):
                global current_recommendations
                current_recommendations = recommendations
                original_send(recommendations)
            
            self.processor.send_game_adaptations = enhanced_send
            
            system_status["running"] = True
            print("✓ Myonix system fully operational!")
            print("\nSystem Components:")
            print("- ESP32 sensor data collection: ACTIVE")
            print("- ML-based exercise state classification: ACTIVE") 
            print("- Real-time game adaptation: ACTIVE")
            print("- React game integration API: http://localhost:5000")
            
            return True
        else:
            print("✗ Failed to connect to ESP32")
            return False
    
    def stop_system(self):
        """Stop the Myonix system"""
        global system_status
        self.processor.is_connected = False
        system_status["running"] = False
        system_status["connected"] = False
        print("Myonix system stopped")

def main():
    """Main function to run Myonix system"""
    system = MyonixSystemManager()
    
    print("=== MYONIX REHABILITATION SYSTEM ===")
    print("Gamified AI-Assisted Adaptive Neuro-Rehab")
    print()
    
    # Replace with your ESP32's Bluetooth MAC address
    ESP32_MAC = "30:AE:A4:XX:XX:XX"  # Find this in your ESP32 serial monitor
    
    if system.start_complete_system(ESP32_MAC):
        try:
            print("\nPress Ctrl+C to stop the system")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            system.stop_system()
            print("\nSystem shutdown complete")
    else:
        print("Failed to start Myonix system")

if __name__ == "__main__":
    main()