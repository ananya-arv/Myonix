#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

// Analog input pins for EMG and EDA
#define EMG_PIN 33
#define EDA_PIN 34

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  delay(1000);
  Serial.println("Initializing sensors...");

  // Initialize I2C for ESP32 Wrover
  Wire.begin(21, 22);

  // Initialize MPU6050
  if (!mpu.begin(0x68, &Wire)) {
    Serial.println("Failed to find MPU6050 chip!");
    while (1) delay(10);
  }

  // Configure MPU6050
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  Serial.println("MPU6050 initialized successfully!");
  Serial.println("---------------------------------------------------");
}

void loop() {
  // === Read IMU Data ===
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // === Read EMG and EDA analog data ===
  int emgValue = analogRead(EMG_PIN);
  int edaValue = analogRead(EDA_PIN);

  // === Print all sensor readings ===
  Serial.println("---------------------------------------------------");
  Serial.print("Acceleration X: "); Serial.print(a.acceleration.x, 3); Serial.println(" m/s^2");
  Serial.print("Acceleration Y: "); Serial.print(a.acceleration.y, 3); Serial.println(" m/s^2");
  Serial.print("Acceleration Z: "); Serial.print(a.acceleration.z, 3); Serial.println(" m/s^2");

  Serial.print("Gyro X: "); Serial.print(g.gyro.x, 3); Serial.println(" rad/s");
  Serial.print("Gyro Y: "); Serial.print(g.gyro.y, 3); Serial.println(" rad/s");
  Serial.print("Gyro Z: "); Serial.print(g.gyro.z, 3); Serial.println(" rad/s");

  Serial.print("Temperature: "); Serial.print(temp.temperature, 2); Serial.println(" °C");

  Serial.print("EMG: "); Serial.println(emgValue);
  Serial.print("EDA: "); Serial.println(edaValue);

  Serial.println("---------------------------------------------------");
  Serial.println();  // blank line for readability

  delay(100);  // 10 Hz sampling rate (adjust as needed)
}
