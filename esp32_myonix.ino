// ESP32 Code Structure for Multi-Modal Data Collection
#include <WiFi.h>
#include <BluetoothSerial.h>
#include <Wire.h>
#include <MPU6050.h>

// Sensor pin definitions
#define EDA_PIN 36      // Grove GSR sensor (GPIO36/A0)
#define EMG_PIN 39      // MyoWare 2.0 sensor (GPIO39/A3)
#define IMU_SCL 22      // MPU6050 I2C clock
#define IMU_SDA 21      // MPU6050 I2C data

// Sampling configuration
const int SAMPLING_RATE = 10;  // 10 Hz (100ms intervals)
const int SAMPLE_INTERVAL = 1000 / SAMPLING_RATE;  // 100ms

// Data structures
struct SensorData {
    float eda_value;
    float emg_value;
    float accel_x, accel_y, accel_z;
    float gyro_x, gyro_y, gyro_z;
    unsigned long timestamp;
};

// Global variables
BluetoothSerial SerialBT;
MPU6050 mpu;
SensorData currentData;
unsigned long lastSample = 0;

void setup() {
    Serial.begin(115200);
    SerialBT.begin("Myonix_Device");
    
    // Initialize I2C for IMU
    Wire.begin(IMU_SDA, IMU_SCL);
    mpu.initialize();
    
    // Configure ADC pins
    analogReadResolution(12);  // 12-bit resolution (0-4095)
    analogSetAttenuation(ADC_11db);  // 0-3.3V range
    
    Serial.println("Myonix sensors initialized");
}

void loop() {
    unsigned long currentTime = millis();
    
    if (currentTime - lastSample >= SAMPLE_INTERVAL) {
        // Read all sensors
        collectSensorData();
        
        // Send data via Bluetooth
        sendDataToPython();
        
        lastSample = currentTime;
    }
}

void collectSensorData() {
    // Read EDA (Grove GSR)
    currentData.eda_value = analogRead(EDA_PIN) * (3.3 / 4095.0);
    
    // Read EMG (MyoWare 2.0)
    currentData.emg_value = analogRead(EMG_PIN) * (3.3 / 4095.0);
    
    // Read IMU (MPU6050)
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    
    // Convert to standard units
    currentData.accel_x = ax / 16384.0;  // ±2g range
    currentData.accel_y = ay / 16384.0;
    currentData.accel_z = az / 16384.0;
    currentData.gyro_x = gx / 131.0;     // ±250°/s range
    currentData.gyro_y = gy / 131.0;
    currentData.gyro_z = gz / 131.0;
    
    currentData.timestamp = millis();
}

void sendDataToPython() {
    // Send JSON formatted data
    SerialBT.print("{");
    SerialBT.print("\"timestamp\":");
    SerialBT.print(currentData.timestamp);
    SerialBT.print(",\"eda\":");
    SerialBT.print(currentData.eda_value, 3);
    SerialBT.print(",\"emg\":");
    SerialBT.print(currentData.emg_value, 3);
    SerialBT.print(",\"accel_x\":");
    SerialBT.print(currentData.accel_x, 3);
    SerialBT.print(",\"accel_y\":");
    SerialBT.print(currentData.accel_y, 3);
    SerialBT.print(",\"accel_z\":");
    SerialBT.print(currentData.accel_z, 3);
    SerialBT.print(",\"gyro_x\":");
    SerialBT.print(currentData.gyro_x, 3);
    SerialBT.print(",\"gyro_y\":");
    SerialBT.print(currentData.gyro_y, 3);
    SerialBT.print(",\"gyro_z\":");
    SerialBT.print(currentData.gyro_z, 3);
    SerialBT.println("}");
}
