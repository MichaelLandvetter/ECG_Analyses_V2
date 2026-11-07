/*
  ESP32-S3 USB ECG stream test sketch

  Wiring:
  - ECG board analog OUT -> ADS1115 A1
  - ADS1115 SDA/SCL -> Baguette S3 STEMMA QT SDA/SCL (GPIO47/GPIO48)
  - ECG board, ADS1115, and Baguette S3 grounds connected together
  - AD8232 LO- -> GPIO3; LO+ -> GPIO4

  USB serial output format:
  timestamp_ms,sample_index,raw_ecg
  Example:
  1234,567,16384
*/

#include <Wire.h>
#include <Adafruit_ADS1X15.h>

static constexpr uint32_t SAMPLE_RATE_HZ = 250;
static constexpr int I2C_SDA_PIN = 47;
static constexpr int I2C_SCL_PIN = 48;
static constexpr uint8_t ADS1115_I2C_ADDRESS = 0x48;
static constexpr uint8_t ADS1115_ECG_CHANNEL = 1;
static constexpr int AD8232_LO_MINUS_PIN = 3;
static constexpr int AD8232_LO_PLUS_PIN = 4;
static constexpr uint32_t BAUD_RATE = 230400;   // Stable default for CSV streaming

static constexpr uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;

Adafruit_ADS1115 ads;
uint32_t sampleIndex = 0;
uint32_t nextSampleDeadlineUs = 0;

void handleIdentification() {
  static char command[32];
  static size_t length = 0;
  static bool overflow = false;
  while (Serial.available() > 0) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\n') {
      command[length] = '\0';
      static constexpr char response[] = "VER_VA,1,ESP32S3,ADS1115,250\n";
      if (!overflow && strcmp(command, "VER_VA_IDENTIFY") == 0 &&
          Serial.availableForWrite() >= static_cast<int>(sizeof(response) - 1)) {
        Serial.write(reinterpret_cast<const uint8_t*>(response), sizeof(response) - 1);
      }
      length = 0;
      overflow = false;
    } else if (c != '\r') {
      if (length < sizeof(command) - 1) {
        command[length++] = c;
      } else {
        overflow = true;
      }
    }
  }
}

void setup() {
  Serial.begin(BAUD_RATE);
  Serial.setTxTimeoutMs(0);  // never block the sampling loop when the PC is slow to read

  const uint32_t serialWaitStartMs = millis();
  while (!Serial && (millis() - serialWaitStartMs) < 2000UL) {
    delay(10);
  }

  pinMode(AD8232_LO_MINUS_PIN, INPUT);
  pinMode(AD8232_LO_PLUS_PIN, INPUT);

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(400000);

  if (!ads.begin(ADS1115_I2C_ADDRESS, &Wire)) {
    Serial.println("Failed to initialize ADS1115 at I2C address 0x48");
    while (true) {
      delay(10);
    }
  }

  ads.setGain(GAIN_ONE);
  ads.setDataRate(RATE_ADS1115_860SPS);

  nextSampleDeadlineUs = micros();
}

void loop() {
  handleIdentification();
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
  const bool leadsOff =
    digitalRead(AD8232_LO_MINUS_PIN) == HIGH ||
    digitalRead(AD8232_LO_PLUS_PIN) == HIGH;
  const int16_t rawEcg = leadsOff ? 0 : ads.readADC_SingleEnded(ADS1115_ECG_CHANNEL);

  char lineBuffer[48];
  const int bytesWritten = snprintf(
    lineBuffer,
    sizeof(lineBuffer),
    "%lu,%lu,%d\n",
    static_cast<unsigned long>(timestampMs),
    static_cast<unsigned long>(sampleIndex),
    static_cast<int>(rawEcg)
  );

  // Skip the line if USB is backed up; the gap shows up as dropped samples on the PC.
  if (bytesWritten > 0 && Serial.availableForWrite() >= bytesWritten) {
    Serial.write(reinterpret_cast<const uint8_t*>(lineBuffer), static_cast<size_t>(bytesWritten));
  }

  ++sampleIndex;
}
