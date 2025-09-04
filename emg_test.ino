int emgPin = 34;  // MyoWare ENV -> GPIO34
int emgValue = 0;

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);  // 12-bit ADC
  analogSetAttenuation(ADC_11db);  // 0-3.3V range
}

void loop() {
  emgValue = analogRead(emgPin);  // 0 - 4095
  float voltage = raw * (3.3 / 4095.0);
  Serial.printf("EMG: %d (%.3fV)\n", raw, voltage);
}
