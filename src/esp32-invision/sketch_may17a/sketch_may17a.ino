// =========================
// LIBRARIES
// =========================
#include <Arduino.h>

// =========================
// UART2 CONFIGURATION
// =========================
HardwareSerial UART2(2);

// UART Pins
#define RXD2 34
#define TXD2 35

// =========================
// ACTUATORS / MOTORS
// =========================
#define MOTOR1 23
#define MOTOR2 22

void setup()
{
  // Initialize primary USB serial port for local debugging console monitor
  Serial.begin(115200);

  // =========================
  // INITIALIZE HARDWARE UART2
  // =========================
  UART2.begin(115200, SERIAL_8N1, RXD2, TXD2);

  // =========================
  // CONFIGURE ESP32 PWM (API v3.x style)
  // =========================
    
  // ledcAttach(pin, frequency, resolution_bits) automatically handles internal channel routing
  ledcAttach(MOTOR1, 1000, 10); // Target Pin, Frequency (1kHz), Resolution (10 bits: value range 0-1023)
  ledcAttach(MOTOR2, 1000, 10); // Target Pin, Frequency (1kHz), Resolution (10 bits: value range 0-1023)

  // Ensure motors are completely shut down on initial boot sequence
  ledcWrite(MOTOR1, 0); 
  ledcWrite(MOTOR2, 0);

  Serial.println("System initialized. Awaiting inbound UART state streams...");
}

void loop()
{
  // Check if state change characters have arrived from the master Raspberry Pi controller
  if (UART2.available() > 0)
    {
      // Read a single raw tracking byte instead of waiting on a newline delimiter string block
      char receivedChar = UART2.read();

      Serial.print("Data Stream Received: ");
      Serial.println(receivedChar);

      // =========================
      // STATE 1: VIBRATION MOTOR ACTIVATION
      // =========================
      if (receivedChar == '1')
        {
	  Serial.println("🚨 [CRITICAL ALERT STATE] Activating Haptic Feedback - VIBRATION ON");
	  ledcWrite(MOTOR1, 1023); // Force full duty-cycle execution across the 10-bit hardware register
	  ledcWrite(MOTOR2, 1023);
        }

      // =========================
      // STATE 0: VIBRATION MOTOR DEACTIVATION
      // =========================
      else if (receivedChar == '0')
        {
	  Serial.println("✅ [PATH CLEAR STATE] Safety Restored - VIBRATION OFF");
	  ledcWrite(MOTOR1, 0); // Drop output voltage immediately back to ground potential
	  ledcWrite(MOTOR2, 0);
        }
    }

  // Tight execution cycle delay to maximize system real-time spatial responsiveness
  delay(10); 
}
