/*
  ESP32-S3 USB ECG stream test sketch

  Wiring:
  - SparkFun analog ECG OUT -> ESP32-S3 GPIO2 (ADC1)
  - ECG board GND -> ESP32-S3 GND
  - ECG board VCC -> board-appropriate supply

  USB serial output format:
  timestamp_ms,sample_index,raw_ecg
  Example:
  1234,567,2048
*/

static constexpr uint32_t SAMPLE_RATE_HZ = 250;
static constexpr int ADC_PIN = 2;               // GPIO2 / ADC1
static constexpr uint32_t BAUD_RATE = 230400;   // Stable default for CSV streaming
static constexpr uint32_t ADC_RESOLUTION_BITS = 12;

static constexpr uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;

uint32_t sampleIndex = 0;
uint32_t nextSampleDeadlineUs = 0;

void setup() {
  Serial.begin(BAUD_RATE);
  analogReadResolution(ADC_RESOLUTION_BITS);
  pinMode(ADC_PIN, INPUT);

  const uint32_t serialWaitStartMs = millis();
  while (!Serial && (millis() - serialWaitStartMs) < 2000UL) {
    delay(10);
  }

  nextSampleDeadlineUs = micros();
}

void loop() {
  const uint32_t nowUs = micros();
  const int32_t timeUntilSampleUs = static_cast<int32_t>(nextSampleDeadlineUs - nowUs);
  if (timeUntilSampleUs > 0) {
    return;
  }

  nextSampleDeadlineUs += SAMPLE_PERIOD_US;
  if (static_cast<int32_t>(nowUs - nextSampleDeadlineUs) > static_cast<int32_t>(SAMPLE_PERIOD_US * 4UL)) {
    nextSampleDeadlineUs = nowUs + SAMPLE_PERIOD_US;
  }

  const uint32_t timestampMs = millis();
  const int rawEcg = analogRead(ADC_PIN);

  char lineBuffer[48];
  const int bytesWritten = snprintf(
    lineBuffer,
    sizeof(lineBuffer),
    "%lu,%lu,%d\n",
    static_cast<unsigned long>(timestampMs),
    static_cast<unsigned long>(sampleIndex),
    rawEcg
  );

  if (bytesWritten > 0) {
    Serial.write(reinterpret_cast<const uint8_t*>(lineBuffer), static_cast<size_t>(bytesWritten));
  }

  ++sampleIndex;
}
