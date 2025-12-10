/*
 * Embedded Device: Event Scheduler System
 *
 * Receives beat prediction events via MQTT and executes precise LED lighting
 * events using hardware timer interrupts on Arduino Nano ESP32.
 *
 * Architecture:
 * - Core 0: Communication (WiFi, MQTT, SNTP)
 * - Core 1: Execution (Event scheduler, timer, ISR)
 *
 * Testing Features:
 * - Latency and time sync testing can be enabled by defining ENABLE_LATENCY_TEST
 * - To enable: Add "-DENABLE_LATENCY_TEST" to build_flags in platformio.ini
 * - When enabled, Arduino will respond to test requests on topics:
 *   - beat/test/latency_request
 *   - beat/test/time_sync_request
 */

#include <Arduino.h>
#include <WiFi.h>
#include <ArduinoJson.h>
#include <time.h>
#include <sys/time.h>
#include "esp_sntp.h"
#include "mqtt_client.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>
#include <freertos/semphr.h>
#include "driver/gpio.h"
#include "wifi_config.h" // WiFi credentials and MQTT config

// ============================================================================
// Configuration
// ============================================================================

// WiFi credentials (from wifi_config.h - macros are used directly)
// MQTT configuration (from wifi_config.h - macros are used directly)

// MQTT Topics
const char *TOPIC_EVENTS_SCHEDULE = "beat/events/schedule";
const char *TOPIC_TIME_SYNC = "beat/time/sync";
const char *TOPIC_COMMANDS = "beat/commands/all";
const char *TOPIC_EXECUTION_LOG = "beat/events/execution_log";

#ifdef ENABLE_LATENCY_TEST
// Test topics (only compiled if ENABLE_LATENCY_TEST is defined)
const char *TOPIC_LATENCY_REQUEST = "beat/test/latency_request";
const char *TOPIC_LATENCY_RESPONSE = "beat/test/latency_response";
const char *TOPIC_TIME_SYNC_REQUEST = "beat/test/time_sync_request";
const char *TOPIC_TIME_SYNC_RESPONSE = "beat/test/time_sync_response";
#endif

// SNTP configuration
// Using local NTP server (chrony on laptop) for millisecond precision
// Fallback to internet server if local server unavailable
const char *TIMEZONE = "UTC"; // Use UTC for consistency

// Enhanced SNTP configuration
#define SNTP_SYNC_INTERVAL_MS 1000 // Sync every 1 second for ms precision
#define SNTP_SYNC_MODE_SMOOTH true // Use smooth sync mode for gradual adjustment

// Event queue configuration
#define MAX_EVENT_QUEUE_SIZE 50
#define EVENT_QUEUE_TIMEOUT_MS 1000

// LED flash duration (milliseconds)
#define LED_FLASH_DURATION_MS 150

// Logging queue configuration
#define MAX_LOG_QUEUE_SIZE 30

// Hardware timer configuration
// Using ESP32 Arduino timer library (hw_timer)

// ============================================================================
// LED Pin Definitions
// ============================================================================

const int LED_BUILTIN_PIN = LED_BUILTIN;

const int LED_RED_PIN = LED_RED;     // RGB Red (GPIO 14)
const int LED_GREEN_PIN = LED_GREEN; // RGB Green (GPIO 15)
const int LED_BLUE_PIN = LED_BLUE;   // RGB Blue (GPIO 16)

// ============================================================================
// Data Structures
// ============================================================================

struct ScheduledEvent
{
    unsigned long execute_time_us; // Microsecond precision
    bool red;
    bool green;
    bool blue;
    String event_id;
};

struct TimeSyncState
{
    bool synced;
    time_t sync_epoch;
    long sync_epoch_usec; // Microseconds part of sync epoch
    unsigned long sync_micros;
    unsigned long time_offset_us;
    // Sync quality monitoring
    unsigned long last_sync_time_ms; // Last sync timestamp (milliseconds since boot)
    int sync_count;                  // Number of successful syncs
    long last_offset_us;             // Last measured offset (for drift tracking)
};

enum EventType
{
    EVENT_TYPE_ON = 0,
    EVENT_TYPE_OFF = 1
};

struct LogEntry
{
    String event_id;
    EventType event_type; // EVENT_TYPE_ON or EVENT_TYPE_OFF
    bool is_automatic;    // true for automatic 150ms turn-off
    unsigned long scheduled_time_us;
    unsigned long actual_time_us;
    time_t scheduled_unix_sec;
    long scheduled_unix_usec;
    time_t actual_unix_sec;
    long actual_unix_usec;
};

// ============================================================================
// Global State
// ============================================================================

// Time synchronization
TimeSyncState timeSync = {false, 0, 0, 0, 0, 0, 0, 0};
SemaphoreHandle_t timeSyncMutex = NULL;

// Event queue (thread-safe)
QueueHandle_t eventQueue = NULL;
SemaphoreHandle_t eventQueueMutex = NULL;
ScheduledEvent eventList[MAX_EVENT_QUEUE_SIZE];
size_t eventCount = 0;

// Hardware timer
// Note: ESP32 Arduino uses hw_timer_t from ESP32 timer library
// We'll use the timer interrupt functionality
volatile ScheduledEvent *nextEvent = NULL;
volatile bool eventExecuted = false;
volatile unsigned long timerAlarmTime = 0;

// MQTT client
esp_mqtt_client_handle_t mqttClient = NULL;
bool mqttConnected = false;

// WiFi status
bool wifiConnected = false;

// Logging queue (thread-safe)
QueueHandle_t logQueue = NULL;

// ============================================================================
// LED Control Functions
// ============================================================================

void setLED(int pin, bool state)
{
    // Regular LEDs: HIGH = ON, LOW = OFF
    digitalWrite(pin, state ? HIGH : LOW);
}

void setRGBLED(int pin, bool state)
{
    // RGB LED is active-low: LOW = ON, HIGH = OFF
    digitalWrite(pin, state ? LOW : HIGH);
}

void initLEDs()
{
    pinMode(LED_BUILTIN_PIN, OUTPUT);
    pinMode(LED_RED_PIN, OUTPUT);
    pinMode(LED_GREEN_PIN, OUTPUT);
    pinMode(LED_BLUE_PIN, OUTPUT);

    // Turn off all LEDs initially
    setLED(LED_BUILTIN_PIN, false);
    setRGBLED(LED_RED_PIN, false);
    setRGBLED(LED_GREEN_PIN, false);
    setRGBLED(LED_BLUE_PIN, false);
}

// ============================================================================
// Time Synchronization
// ============================================================================

void timeSyncNotificationCallback(struct timeval *tv)
{
    // Called when SNTP syncs time
    unsigned long current_ms = millis();

    if (xSemaphoreTake(timeSyncMutex, portMAX_DELAY))
    {
        // Calculate time since last sync (for drift tracking)
        unsigned long time_since_last_sync = 0;
        if (timeSync.synced && timeSync.last_sync_time_ms > 0)
        {
            time_since_last_sync = current_ms - timeSync.last_sync_time_ms;
        }

        // Store previous offset for comparison
        long previous_offset = timeSync.last_offset_us;

        // Update sync state
        timeSync.sync_epoch = tv->tv_sec;
        timeSync.sync_epoch_usec = tv->tv_usec;
        timeSync.sync_micros = micros();
        timeSync.synced = true;
        timeSync.time_offset_us = 0; // Will be calculated on first event
        timeSync.last_sync_time_ms = current_ms;
        timeSync.sync_count++;

        // Calculate offset (for monitoring - this is approximate)
        timeSync.last_offset_us = 0; // Will be updated when we have reference

        xSemaphoreGive(timeSyncMutex);

        // Log sync information
        Serial.printf("Time synchronized via SNTP (sync #%d", timeSync.sync_count);
        if (time_since_last_sync > 0)
        {
            Serial.printf(", %lu ms since last sync", time_since_last_sync);
        }
        Serial.println(")");
    }
}

void initSNTP()
{
    Serial.println("Initializing Enhanced SNTP with local NTP server...");

    // Print local IP address for reference
    Serial.print("Local IP address: ");
    Serial.println(WiFi.localIP());
    Serial.print("NTP Server (local): ");
    Serial.println(NTP_SERVER_LOCAL);
    Serial.print("NTP Server (fallback): ");
    Serial.println(NTP_SERVER_FALLBACK);

    // Configure SNTP operating mode
    sntp_setoperatingmode(SNTP_OPMODE_POLL);

    // Set NTP servers: local server first, fallback second
    sntp_setservername(0, NTP_SERVER_LOCAL);
    // sntp_setservername(1, NTP_SERVER_FALLBACK);

    // Set sync notification callback
    sntp_set_time_sync_notification_cb(timeSyncNotificationCallback);

// Enhanced SNTP configuration for millisecond precision
// Note: These functions may not be available in all ESP-IDF versions
// If compilation fails, comment out these lines - the local server IP is the key improvement

// Set sync interval to 1 second for frequent polling (if available)
#ifdef sntp_set_sync_interval
    sntp_set_sync_interval(SNTP_SYNC_INTERVAL_MS);
    Serial.printf("  Sync interval set to %d ms\n", SNTP_SYNC_INTERVAL_MS);
#else
    // Function not available - SNTP will use default interval (typically 1 hour)
    // Frequent syncs will still occur due to local server low latency
    Serial.println("  Note: sntp_set_sync_interval not available, using default");
#endif

// Use smooth sync mode for gradual time adjustment (reduces jitter)
#if defined(SNTP_SYNC_MODE_SMOOTH) && defined(sntp_set_sync_mode)
    sntp_set_sync_mode(SNTP_SYNC_MODE_SMOOTH);
    Serial.println("  Smooth sync mode enabled");
#else
    Serial.println("  Note: Smooth sync mode not available, using default");
#endif

    // Initialize SNTP
    sntp_init();

    Serial.println("SNTP initialization complete");

    // Set timezone (UTC for consistency)
    setenv("TZ", TIMEZONE, 1);
    tzset();

    // Wait for initial sync
    // Note: SNTP sync is asynchronous - the callback will fire when complete
    Serial.println("Waiting for SNTP time sync...");
    int retries = 0;

    // Wait for either status to complete OR callback to set synced flag
    while (retries < 60)
    { // Wait up to 30 seconds (60 * 500ms)
        // Check if callback has already set the synced flag
        if (xSemaphoreTake(timeSyncMutex, 0) == pdTRUE)
        {
            bool synced = timeSync.synced;
            xSemaphoreGive(timeSyncMutex);
            if (synced)
            {
                Serial.println("Time sync successful! (via callback)");
                return; // Callback already handled everything
            }
        }

        // Also check SNTP status
        if (sntp_get_sync_status() == SNTP_SYNC_STATUS_COMPLETED)
        {
            struct timeval tv;
            gettimeofday(&tv, NULL);

            if (xSemaphoreTake(timeSyncMutex, portMAX_DELAY))
            {
                unsigned long current_ms = millis();
                timeSync.sync_epoch = tv.tv_sec;
                timeSync.sync_epoch_usec = tv.tv_usec;
                timeSync.sync_micros = micros();
                timeSync.synced = true;
                timeSync.time_offset_us = 0;
                timeSync.last_sync_time_ms = current_ms;
                timeSync.sync_count++;
                timeSync.last_offset_us = 0;
                xSemaphoreGive(timeSyncMutex);
            }

            Serial.printf("Time sync successful! (via status check, sync #%d)\n", timeSync.sync_count);
            return;
        }

        delay(500);
        retries++;
    }

    // Final check - callback might have fired while we were waiting
    if (xSemaphoreTake(timeSyncMutex, 0) == pdTRUE)
    {
        bool synced = timeSync.synced;
        xSemaphoreGive(timeSyncMutex);
        if (synced)
        {
            Serial.println("Time sync successful! (callback fired during wait)");
            return;
        }
    }

    Serial.println("Time sync timeout - will continue and retry in background");
    Serial.println("SNTP will automatically retry periodically");
}

// Forward declaration
void logSyncQuality();

// Log sync quality information
void logSyncQuality()
{
    if (xSemaphoreTake(timeSyncMutex, 100 / portTICK_PERIOD_MS) == pdTRUE)
    {
        if (timeSync.synced)
        {
            struct timeval tv;
            gettimeofday(&tv, NULL);

            // Calculate time since last sync
            unsigned long time_since_sync = 0;
            if (timeSync.last_sync_time_ms > 0)
            {
                time_since_sync = millis() - timeSync.last_sync_time_ms;
            }

            Serial.printf("Sync Quality: synced=%d, sync_count=%d, time_since_sync=%lu ms\n",
                          timeSync.synced, timeSync.sync_count, time_since_sync);
            Serial.printf("Current time: %ld.%06ld, sync_epoch: %ld.%06ld\n",
                          tv.tv_sec, tv.tv_usec, timeSync.sync_epoch, timeSync.sync_epoch_usec);
        }
        else
        {
            Serial.println("Sync Quality: Not synced");
        }
        xSemaphoreGive(timeSyncMutex);
    }
}

unsigned long unixTimeToMicros(time_t unixTime, long microseconds)
{
    if (!timeSync.synced)
    {
        Serial.println("WARNING: Time not synced, using relative timing");
        return micros() + (microseconds / 1000); // Fallback to relative
    }

    if (xSemaphoreTake(timeSyncMutex, portMAX_DELAY))
    {
        // Calculate difference from SYNC POINT (not current time)
        long diffSeconds = unixTime - timeSync.sync_epoch;
        long diffMicros = microseconds - timeSync.sync_epoch_usec;

        // Convert to micros() equivalent using sync point as reference
        unsigned long executeTime = timeSync.sync_micros + (diffSeconds * 1000000L) + diffMicros;

        xSemaphoreGive(timeSyncMutex);
        Serial.printf("Sync: epoch=%ld, epoch_usec=%ld, sync_micros=%lu\n",
                      timeSync.sync_epoch, timeSync.sync_epoch_usec, timeSync.sync_micros);
        Serial.printf("Diff: seconds=%ld, micros=%ld, executeTime=%lu\n", diffSeconds, diffMicros, executeTime);
        return executeTime;
    }
    Serial.printf("Using current micros fallback timing\n");
    return micros(); // Fallback
}

// Convert micros() value to Unix timestamp
void microsToUnixTime(unsigned long microsValue, time_t *unix_sec, long *unix_usec)
{
    if (!timeSync.synced)
    {
        *unix_sec = 0;
        *unix_usec = 0;
        return;
    }

    if (xSemaphoreTake(timeSyncMutex, portMAX_DELAY))
    {
        // Calculate difference from sync point
        // Handle potential wraparound (micros() wraps after ~70 minutes)
        unsigned long diffMicros;
        if (microsValue >= timeSync.sync_micros)
        {
            diffMicros = microsValue - timeSync.sync_micros;
        }
        else
        {
            // Wraparound occurred - assume it's been less than 70 minutes
            diffMicros = microsValue + (0xFFFFFFFF - timeSync.sync_micros);
        }

        // Convert to seconds and microseconds
        long diffSeconds = diffMicros / 1000000L;
        long diffMicrosRemainder = diffMicros % 1000000L;

        // Add to sync epoch (with microseconds)
        *unix_sec = timeSync.sync_epoch + diffSeconds;
        *unix_usec = timeSync.sync_epoch_usec + diffMicrosRemainder;

        // Handle overflow in microseconds
        if (*unix_usec >= 1000000L)
        {
            *unix_sec += 1;
            *unix_usec -= 1000000L;
        }
        else if (*unix_usec < 0)
        {
            // Handle underflow
            *unix_sec -= 1;
            *unix_usec += 1000000L;
        }

        xSemaphoreGive(timeSyncMutex);
    }
    else
    {
        *unix_sec = 0;
        *unix_usec = 0;
    }
}

// ============================================================================
// Event Queue Management
// ============================================================================

bool insertEventSorted(ScheduledEvent event)
{
    if (xSemaphoreTake(eventQueueMutex, EVENT_QUEUE_TIMEOUT_MS / portTICK_PERIOD_MS) != pdTRUE)
    {
        Serial.println("ERROR: Failed to acquire event queue mutex");
        return false;
    }

    if (eventCount >= MAX_EVENT_QUEUE_SIZE)
    {
        Serial.println("WARNING: Event queue full, rejecting event");
        xSemaphoreGive(eventQueueMutex);
        return false;
    }

    // Insert event in sorted order (by execute_time_us)
    size_t insertIndex = eventCount;
    for (size_t i = 0; i < eventCount; i++)
    {
        if (event.execute_time_us < eventList[i].execute_time_us)
        {
            insertIndex = i;
            break;
        }
    }

    // Shift events to make room
    for (size_t i = eventCount; i > insertIndex; i--)
    {
        eventList[i] = eventList[i - 1];
    }

    // Insert new event
    eventList[insertIndex] = event;
    eventCount++;

    xSemaphoreGive(eventQueueMutex);
    return true;
}

bool removeEvent(size_t index)
{
    if (xSemaphoreTake(eventQueueMutex, EVENT_QUEUE_TIMEOUT_MS / portTICK_PERIOD_MS) != pdTRUE)
    {
        return false;
    }

    if (index >= eventCount)
    {
        xSemaphoreGive(eventQueueMutex);
        return false;
    }

    // Shift events to fill gap
    for (size_t i = index; i < eventCount - 1; i++)
    {
        eventList[i] = eventList[i + 1];
    }

    eventCount--;
    xSemaphoreGive(eventQueueMutex);
    return true;
}

ScheduledEvent *peekNextEvent()
{
    if (eventCount == 0)
    {
        return NULL;
    }
    return &eventList[0];
}

// ============================================================================
// Logging Functions
// ============================================================================

bool enqueueLogEntry(LogEntry entry)
{
    if (logQueue == NULL)
    {
        return false;
    }

    // Try to send to queue (non-blocking)
    BaseType_t result = xQueueSend(logQueue, &entry, 0);

    if (result != pdTRUE)
    {
        // Queue full - drop newest entries by receiving and discarding until we can send
        LogEntry dummy;
        while (xQueueReceive(logQueue, &dummy, 0) == pdTRUE)
        {
            // Discard oldest entries to make room
        }
        // Now try to send again
        result = xQueueSend(logQueue, &entry, 0);
    }

    return (result == pdTRUE);
}

// ============================================================================
// Hardware Timer and ISR
// ============================================================================

// For Arduino ESP32, we'll use a simpler approach: check timer in scheduler task
// For more precise timing, we could use hardware timer interrupts, but Arduino
// ESP32 timer API is different. For now, we'll use micros() polling with
// hardware timer checking in the scheduler task.

void executeEvent(ScheduledEvent *event)
{
    if (event == NULL)
    {
        return;
    }

        // Determine event type and if it's automatic
    bool is_turn_on = (event->red || event->green || event->blue);
    bool is_automatic = (event->event_id == "0"); // event_id=0 means automatic turn-off

    // Set LED states
    Serial.printf("Setting LEDs: R=%d (pin %d), G=%d (pin %d), B=%d (pin %d)\n",
                  event->red, LED_RED_PIN, event->green, LED_GREEN_PIN,
                  event->blue, LED_BLUE_PIN);
    setRGBLED(LED_RED_PIN, event->red);
    setRGBLED(LED_GREEN_PIN, event->green);
    setRGBLED(LED_BLUE_PIN, event->blue);
    unsigned long actual_time_us = micros();
    Serial.printf("Event executed: ID=%d, RGB=(%d,%d,%d)\n",
                  event->event_id, event->red, event->green, event->blue);

    // Create log entry
    LogEntry logEntry;
    logEntry.event_id = event->event_id;
    logEntry.event_type = is_turn_on ? EVENT_TYPE_ON : EVENT_TYPE_OFF;
    logEntry.is_automatic = is_automatic;
    logEntry.scheduled_time_us = event->execute_time_us;
    logEntry.actual_time_us = actual_time_us;

    // Convert scheduled time to Unix timestamp
    microsToUnixTime(event->execute_time_us, &logEntry.scheduled_unix_sec, &logEntry.scheduled_unix_usec);

    // Convert actual time to Unix timestamp
    microsToUnixTime(actual_time_us, &logEntry.actual_unix_sec, &logEntry.actual_unix_usec);

    // Enqueue log entry (non-blocking, drops newest if queue full)
    enqueueLogEntry(logEntry);
}

void initHardwareTimer()
{
    // Initialize GPIO pins for fast access
    gpio_config_t io_conf = {};
    io_conf.pin_bit_mask = ((1ULL << LED_RED_PIN) | (1ULL << LED_GREEN_PIN) | (1ULL << LED_BLUE_PIN));
    io_conf.mode = GPIO_MODE_OUTPUT;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.intr_type = GPIO_INTR_DISABLE;
    gpio_config(&io_conf);

    Serial.println("GPIO configured for fast LED control");
}

void configureTimerForEvent(ScheduledEvent *event)
{
    if (event == NULL)
    {
        return;
    }

    // Simply set the next event - scheduler will check timing
    nextEvent = event;
    timerAlarmTime = event->execute_time_us;
}

// ============================================================================
// MQTT Message Handling
// ============================================================================

void scheduleSingleEvent(JsonObject &event)
{
    time_t unix_time = event["unix_time"].as<time_t>();
    long microseconds = event["microseconds"].as<long>();
    String event_id_str = event["event_id"].as<String>();
    String event_type_str = "";
    float event_sub_id = 0.0;
    int underscore_idx = event_id_str.indexOf('_');
    if (underscore_idx >= 0)
    {
        event_type_str = event_id_str.substring(0, underscore_idx);
        String float_part = event_id_str.substring(underscore_idx + 1);
        event_sub_id = float_part.toFloat();
    }
    else
    {
        event_type_str = event_id_str;
        event_sub_id = 0.0;
    }
    bool red = (event["r"].as<int>() != 0);
    bool green = (event["g"].as<int>() != 0);
    bool blue = (event["b"].as<int>() != 0);

    Serial.printf("Event: ID=%s, Type=%s, Sub-ID=%f\n", event_id_str.c_str(), event_type_str.c_str(), event_sub_id);

    // Get the current synchronized Unix time and microseconds
    time_t current_unix_sec;
    long current_unix_usec;
    microsToUnixTime(micros(), &current_unix_sec, &current_unix_usec);
    Serial.printf("Current Unix time: %ld, microseconds: %ld\n", current_unix_sec, current_unix_usec);
    Serial.printf("Event Unix time: %ld, microseconds: %ld\n", unix_time, microseconds);

    // Convert event time (center of flash) to micros() equivalent
    unsigned long event_time_us = unixTimeToMicros(unix_time, microseconds);

    Serial.printf("Event: Unix Time=%ld, Microseconds=%ld, Event Time=%lu\n", unix_time, microseconds, event_time_us);
    Serial.printf("Current micros: %lu\n", micros());

    // Calculate turn-on and turn-off times centered around event time
    unsigned long flash_duration_half_us = (LED_FLASH_DURATION_MS * 1000UL) / 2;
    unsigned long turn_on_time_us = event_time_us - flash_duration_half_us;
    unsigned long turn_off_time_us = event_time_us + flash_duration_half_us;

    unsigned long current_micros = micros();

    // Handle edge case: if turn-on time is in the past
    bool turn_on_immediately = false;
    if (turn_on_time_us < current_micros)
    {
        Serial.printf("Event turn-on time in past, skipping: ID=%s\n", event_id_str.c_str());
        return;
    }

    // If turn-off time is also in the past, skip scheduling
    if (turn_off_time_us < current_micros)
    {
        Serial.printf("Event turn-off time in past, skipping: ID=%s\n", event_id_str.c_str());
        return;
    }

    // Schedule turn-on event (if not already executed immediately)
    if (!turn_on_immediately)
    {
        ScheduledEvent turnOnEvent;
        turnOnEvent.execute_time_us = turn_on_time_us;
        turnOnEvent.red = red;
        turnOnEvent.green = green;
        turnOnEvent.blue = blue;
        turnOnEvent.event_id = event_id_str;

        if (insertEventSorted(turnOnEvent))
        {
            Serial.printf("Turn-on event scheduled: ID=%d, time=%lu, RGB=(%d,%d,%d)\n",
                          turnOnEvent.event_id,
                          turnOnEvent.execute_time_us,
                          turnOnEvent.red, turnOnEvent.green, turnOnEvent.blue);
        }
        else
        {
            Serial.println("Failed to schedule turn-on event");
        }
    }

    // Schedule turn-off event (automatic, so event_id=0)
    ScheduledEvent turnOffEvent;
    turnOffEvent.execute_time_us = turn_off_time_us;
    turnOffEvent.red = false;
    turnOffEvent.green = false;
    turnOffEvent.blue = false;
    turnOffEvent.event_id = "0"; // event_id=0 marks automatic turn-off events

    if (insertEventSorted(turnOffEvent))
    {
        Serial.printf("Turn-off event scheduled: ID=%d (automatic), time=%lu\n",
                      turnOffEvent.event_id,
                      turnOffEvent.execute_time_us);
    }
    else
    {
        Serial.println("Failed to schedule turn-off event");
    }
}

void handleScheduleEvent(JsonDocument &doc)
{
    // Check if batch or single event (using new API)
    if (doc["events"].is<JsonArray>())
    {
        // Batch events
        JsonArray events = doc["events"];
        for (JsonObject event : events)
        {
            scheduleSingleEvent(event);
        }
    }
    else
    {
        // Single event - get JsonObject reference
        JsonObject eventObj = doc.as<JsonObject>();
        scheduleSingleEvent(eventObj);
    }
}

void handleTimeSync(JsonDocument &doc)
{
    time_t unix_time = doc["unix_time"] | 0;
    long microseconds = doc["microseconds"] | 0;

    // Update time sync using received host timestamp (includes microseconds)
    if (xSemaphoreTake(timeSyncMutex, portMAX_DELAY))
    {
        timeSync.sync_epoch = unix_time;
        timeSync.sync_epoch_usec = microseconds;
        timeSync.sync_micros = micros(); // micros() at the moment we apply sync
        timeSync.synced = true;
        timeSync.time_offset_us = 0;
        xSemaphoreGive(timeSyncMutex);
    }

    Serial.printf("Time sync updated via MQTT: %ld.%06ld\n", unix_time, microseconds);
}

#ifdef ENABLE_LATENCY_TEST
// ============================================================================
// Latency and Time Sync Test Handlers
// ============================================================================

void handleLatencyTest(JsonDocument &doc)
{
    int request_id = doc["request_id"] | -1;
    if (request_id < 0)
    {
        Serial.println("ERROR: Invalid latency test request (missing request_id)");
        return;
    }

    // get current time on arduino and Convert to Unix time if synced
    unsigned long current_micros = micros();
    time_t current_unix_sec;
    long current_unix_usec;
    microsToUnixTime(current_micros, &current_unix_sec, &current_unix_usec);

    // Create response JSON
    JsonDocument responseDoc;
    responseDoc["request_id"] = request_id;
    responseDoc["arduino_timestamp_ms"] = 0;
    responseDoc["arduino_timestamp_us"] = current_micros;
    responseDoc["arduino_unix_time"] = current_unix_sec;
    responseDoc["arduino_microseconds"] = current_unix_usec;

    // Serialize response
    char json_buffer[256];
    size_t json_len = serializeJson(responseDoc, json_buffer, sizeof(json_buffer));

    // Publish response
    if (mqttClient != NULL && mqttConnected)
    {
        int msg_id = esp_mqtt_client_publish(
            mqttClient,
            TOPIC_LATENCY_RESPONSE,
            json_buffer,
            json_len,
            1, // QoS 1
            0  // Retain false
        );

        if (msg_id < 0)
        {
            Serial.println("ERROR: Failed to publish latency test response");
        }
    }
}

void handleTimeSyncTest(JsonDocument &doc)
{
    int request_id = doc["request_id"] | -1;
    if (request_id < 0)
    {
        Serial.println("ERROR: Invalid time sync test request (missing request_id)");
        return;
    }

    // Get current synced Unix time directly (not from micros())
    time_t arduino_unix_sec = 0;
    long arduino_unix_usec = 0;
    microsToUnixTime(micros(), &arduino_unix_sec, &arduino_unix_usec);

    // Create response JSON
    JsonDocument responseDoc;
    responseDoc["request_id"] = request_id;
    responseDoc["arduino_unix_time"] = arduino_unix_sec;
    responseDoc["arduino_microseconds"] = arduino_unix_usec;

    // Serialize response
    char json_buffer[256];
    size_t json_len = serializeJson(responseDoc, json_buffer, sizeof(json_buffer));

    // Publish response
    if (mqttClient != NULL && mqttConnected)
    {
        int msg_id = esp_mqtt_client_publish(
            mqttClient,
            TOPIC_TIME_SYNC_RESPONSE,
            json_buffer,
            json_len,
            1, // QoS 1
            0  // Retain false
        );

        if (msg_id < 0)
        {
            Serial.println("ERROR: Failed to publish time sync test response");
        }
    }
}
#endif // ENABLE_LATENCY_TEST

void handleMQTTMessage(const char *topic, int topic_len, const char *data, int data_len)
{
    // Convert to null-terminated strings
    char topic_str[topic_len + 1];
    char data_str[data_len + 1];
    memcpy(topic_str, topic, topic_len);
    topic_str[topic_len] = '\0';
    memcpy(data_str, data, data_len);
    data_str[data_len] = '\0';

    Serial.printf("MQTT message received: topic=%s, len=%d\n", topic_str, data_len);

    // Parse JSON (using JsonDocument for v7 - replaces deprecated StaticJsonDocument)
    // ArduinoJson v7: JsonDocument uses dynamic allocation, which is fine for our small messages
    JsonDocument doc;
    DeserializationError error = deserializeJson(doc, data_str);

    if (error)
    {
        Serial.printf("JSON parse error: %s\n", error.c_str());
        return;
    }

    // Route to appropriate handler
    if (strcmp(topic_str, TOPIC_EVENTS_SCHEDULE) == 0)
    {
        handleScheduleEvent(doc);
    }
    else if (strcmp(topic_str, TOPIC_TIME_SYNC) == 0)
    {
        handleTimeSync(doc);
    }
    else if (strcmp(topic_str, TOPIC_COMMANDS) == 0)
    {
        Serial.println("Command received (not implemented)");
        // Future: handle commands
    }
#ifdef ENABLE_LATENCY_TEST
    else if (strcmp(topic_str, TOPIC_LATENCY_REQUEST) == 0)
    {
        handleLatencyTest(doc);
    }
    else if (strcmp(topic_str, TOPIC_TIME_SYNC_REQUEST) == 0)
    {
        handleTimeSyncTest(doc);
    }
#endif // ENABLE_LATENCY_TEST
}

// ============================================================================
// MQTT Event Handler
// ============================================================================

static void mqtt_event_handler(void *handler_args, esp_event_base_t base,
                               int32_t event_id, void *event_data)
{
    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)event_data;
    esp_mqtt_client_handle_t client = mqttClient;

    if (event == NULL && event_id != MQTT_EVENT_CONNECTED && event_id != MQTT_EVENT_DISCONNECTED)
    {
        return; // Some events don't have data
    }

    switch ((esp_mqtt_event_id_t)event_id)
    {
    case MQTT_EVENT_CONNECTED:
        Serial.println("MQTT Connected");
        mqttConnected = true;

        // Subscribe to topics with QoS 1
        if (client != NULL)
        {
            esp_mqtt_client_subscribe(client, TOPIC_EVENTS_SCHEDULE, 1);
            esp_mqtt_client_subscribe(client, TOPIC_TIME_SYNC, 1);
            esp_mqtt_client_subscribe(client, TOPIC_COMMANDS, 1);
#ifdef ENABLE_LATENCY_TEST
            esp_mqtt_client_subscribe(client, TOPIC_LATENCY_REQUEST, 1);
            esp_mqtt_client_subscribe(client, TOPIC_TIME_SYNC_REQUEST, 1);
#endif // ENABLE_LATENCY_TEST
        }

        Serial.println("Subscribed to MQTT topics");
#ifdef ENABLE_LATENCY_TEST
        Serial.println("  (Test topics enabled)");
#endif // ENABLE_LATENCY_TEST
        break;

    case MQTT_EVENT_DISCONNECTED:
        Serial.println("MQTT Disconnected");
        mqttConnected = false;
        break;

    case MQTT_EVENT_DATA:
        if (event != NULL)
        {
            handleMQTTMessage(
                event->topic,
                event->topic_len,
                event->data,
                event->data_len);
        }
        break;

    case MQTT_EVENT_ERROR:
        Serial.println("MQTT Error");
        break;

    default:
        break;
    }
}

// ============================================================================
// FreeRTOS Tasks
// ============================================================================

// Core 0: MQTT Client Task
void mqttClientTask(void *parameter)
{
    Serial.println("MQTT Client Task started on Core 0");

    while (true)
    {
        if (wifiConnected && !mqttConnected)
        {
            // Initialize MQTT client (ESP32 Arduino uses URI-based config)
            char mqtt_uri[128];
            snprintf(mqtt_uri, sizeof(mqtt_uri), "mqtt://%s:%d", MQTT_BROKER, MQTT_PORT);

            esp_mqtt_client_config_t mqtt_cfg = {};
            mqtt_cfg.uri = mqtt_uri;
            mqtt_cfg.client_id = MQTT_CLIENT_ID;

            mqttClient = esp_mqtt_client_init(&mqtt_cfg);

            // Register event handler (correct signature for ESP32 Arduino)
            esp_mqtt_client_register_event(
                mqttClient,
                MQTT_EVENT_ANY,
                mqtt_event_handler,
                NULL);

            esp_mqtt_client_start(mqttClient);

            Serial.println("MQTT client started");
        }

        // Small delay
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
}

// Core 0: Event Logger Task
void eventLoggerTask(void *parameter)
{
    Serial.println("Event Logger Task started on Core 0");

    unsigned long lastSyncQualityLog = 0;
    const unsigned long SYNC_QUALITY_LOG_INTERVAL_MS = 120000; // Log every 120 seconds

    while (true)
    {
        // Periodically log sync quality
        unsigned long now_ms = millis();
        if (now_ms - lastSyncQualityLog >= SYNC_QUALITY_LOG_INTERVAL_MS)
        {
            logSyncQuality();
            lastSyncQualityLog = now_ms;
        }

        if (logQueue != NULL && mqttConnected && mqttClient != NULL)
        {
            LogEntry logEntry;

            // Try to receive log entry from queue (blocking with timeout)
            if (xQueueReceive(logQueue, &logEntry, 100 / portTICK_PERIOD_MS) == pdTRUE)
            {
                // Convert to JSON
                JsonDocument doc;
                doc["event_id"] = logEntry.event_id;
                doc["event_type"] = (logEntry.event_type == EVENT_TYPE_ON) ? "on" : "off";
                doc["is_automatic"] = logEntry.is_automatic;

                // Format scheduled time as seconds.microseconds
                char scheduled_time_str[32];
                if (logEntry.scheduled_unix_sec > 0)
                {
                    snprintf(scheduled_time_str, sizeof(scheduled_time_str), "%ld.%06ld",
                             logEntry.scheduled_unix_sec, logEntry.scheduled_unix_usec);
                }
                else
                {
                    snprintf(scheduled_time_str, sizeof(scheduled_time_str), "0.000000");
                }
                doc["scheduled_time"] = scheduled_time_str;

                // Format actual time as seconds.microseconds
                char actual_time_str[32];
                if (logEntry.actual_unix_sec > 0)
                {
                    snprintf(actual_time_str, sizeof(actual_time_str), "%ld.%06ld",
                             logEntry.actual_unix_sec, logEntry.actual_unix_usec);
                }
                else
                {
                    snprintf(actual_time_str, sizeof(actual_time_str), "0.000000");
                }
                doc["actual_time"] = actual_time_str;

                // Serialize JSON
                char json_buffer[256];
                size_t json_len = serializeJson(doc, json_buffer, sizeof(json_buffer));

                // Publish to MQTT
                int msg_id = esp_mqtt_client_publish(
                    mqttClient,
                    TOPIC_EXECUTION_LOG,
                    json_buffer,
                    json_len, // Length of data
                    0,        // QoS 0 (fire and forget for logging)
                    0         // Retain false
                );

                if (msg_id < 0)
                {
                    // MQTT publish failed, but we'll continue processing
                    // The entry is already removed from queue, so it's lost
                }
            }
        }
        else
        {
            // Wait a bit if MQTT not ready or queue not initialized
            vTaskDelay(100 / portTICK_PERIOD_MS);
        }
    }
}

// Core 1: Event Scheduler Task
void eventSchedulerTask(void *parameter)
{
    Serial.println("Event Scheduler Task started on Core 1");

    while (true)
    {
        unsigned long now = micros();
        bool eventExecuted = false;

        // 1. Check if current event should be executed
        if (nextEvent != NULL)
        {
            if (now >= timerAlarmTime)
            {
                // Execute event (cast away volatile for function call)
                ScheduledEvent *event = const_cast<ScheduledEvent *>(nextEvent);
                executeEvent(event);
                eventExecuted = true;

                // Remove executed event
                if (eventCount > 0)
                {
                    removeEvent(0);
                }
                nextEvent = NULL;
                timerAlarmTime = 0;
            }
        }

        // 2. Check for past-due events (execute immediately if missed)
        if (eventCount > 0)
        {
            ScheduledEvent *next = peekNextEvent();
            if (next != NULL && next->execute_time_us <= now)
            {
                // Event is past due, execute immediately
                executeEvent(next);
                removeEvent(0);
                eventExecuted = true;
                nextEvent = NULL;
                timerAlarmTime = 0;
            }
        }

        // 3. Check for new events and configure timer (if no event was just executed)
        if (!eventExecuted && eventCount > 0)
        {
            if (nextEvent == NULL)
            {
                // No event scheduled, get the next one
                ScheduledEvent *next = peekNextEvent();
                if (next != NULL)
                {
                    configureTimerForEvent(next);
                }
            }
            else
            {
                // Check if there's an earlier event
                ScheduledEvent *next = peekNextEvent();
                if (next != NULL && next->execute_time_us < timerAlarmTime)
                {
                    configureTimerForEvent(next);
                }
            }
        }

        // Small delay - reduced for better responsiveness
        // Use yield to allow other tasks to run, but check more frequently
        vTaskDelay(1 / portTICK_PERIOD_MS); // 1ms delay (minimum for FreeRTOS)
    }
}

// ============================================================================
// WiFi Setup
// ============================================================================

void setupWiFi()
{
    Serial.print("Connecting to WiFi: ");
    Serial.println(WIFI_SSID);

    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30)
    {
        delay(500);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED)
    {
        wifiConnected = true;
        Serial.println("");
        Serial.println("WiFi connected!");
        Serial.print("IP address: ");
        Serial.println(WiFi.localIP());
    }
    else
    {
        Serial.println("");
        Serial.println("WiFi connection failed");
    }
}

// ============================================================================
// Setup and Main Loop
// ============================================================================

void setup()
{
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n=== Embedded Device: Event Scheduler System ===");

    // Initialize LEDs
    initLEDs();
    Serial.println("LEDs initialized");

    // Create synchronization primitives
    timeSyncMutex = xSemaphoreCreateMutex();
    eventQueueMutex = xSemaphoreCreateMutex();

    if (timeSyncMutex == NULL || eventQueueMutex == NULL)
    {
        Serial.println("ERROR: Failed to create mutexes");
        return;
    }

    // Create logging queue
    logQueue = xQueueCreate(MAX_LOG_QUEUE_SIZE, sizeof(LogEntry));
    if (logQueue == NULL)
    {
        Serial.println("ERROR: Failed to create log queue");
        return;
    }

    // Initialize hardware timer
    initHardwareTimer();

    // Test LEDs to verify functionality
    Serial.println("Testing LEDs...");
    delay(1000);
    setRGBLED(LED_RED_PIN, true);
    delay(1000);
    setRGBLED(LED_RED_PIN, false);
    setRGBLED(LED_GREEN_PIN, true);
    delay(1000);
    setRGBLED(LED_GREEN_PIN, false);
    setRGBLED(LED_BLUE_PIN, true);
    delay(1000);
    setRGBLED(LED_BLUE_PIN, false);
    Serial.println("LED test complete");

    // Setup WiFi
    setupWiFi();

    // Initialize SNTP
    if (wifiConnected)
    {
        initSNTP();

        // Log initial sync quality after a short delay
        delay(2000);
        logSyncQuality();
    }

    // Create FreeRTOS tasks
    // Core 0: MQTT Client Task
    xTaskCreatePinnedToCore(
        mqttClientTask,
        "MQTTClient",
        8192, // Stack size
        NULL,
        2, // Priority
        NULL,
        0 // Core 0
    );

    // Core 0: Event Logger Task
    xTaskCreatePinnedToCore(
        eventLoggerTask,
        "EventLogger",
        4096, // Stack size
        NULL,
        1, // Priority (lower than MQTT client)
        NULL,
        0 // Core 0
    );

    // Core 1: Event Scheduler Task
    xTaskCreatePinnedToCore(
        eventSchedulerTask,
        "EventScheduler",
        4096, // Stack size
        NULL,
        2, // Higher priority for timing-critical
        NULL,
        1 // Core 1
    );

    Serial.println("System initialized - tasks created");
    Serial.println("Waiting for MQTT connection and events...");

    // Delete setup task (tasks will handle everything)
    vTaskDelete(NULL);
}

void loop()
{
    // This should never execute since setup() deletes itself
    // But keep it as a safety net
    vTaskDelay(1000 / portTICK_PERIOD_MS);
}
