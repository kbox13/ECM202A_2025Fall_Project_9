#include "mqtt_publisher.h"
#include "prediction_types.h"
#include "mqtt_command_logger.h"
#include <mqtt/async_client.h>
#include <mqtt/connect_options.h>
#include <mqtt/message.h>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <cstring>
#include <cmath>

namespace essentia {
namespace streaming {

const char* MQTTPublisher::name = "MQTTPublisher";
const char* MQTTPublisher::category = "Streaming";
const char* MQTTPublisher::description =
  "Publishes lighting commands to MQTT broker with Unix timestamp conversion.";

MQTTPublisher::MQTTPublisher() : Algorithm() {
  declareInput(_in, "in", "lighting commands from LightingEngine");
  _in.setAcquireSize(1);
  _in.setReleaseSize(1);
  
  _mqttConnected = false;
  _timeInitialized = false;
  _startTimeSec = 0.0;
  _commandLogger = nullptr;
}

MQTTPublisher::~MQTTPublisher() {
  cleanupMQTT();
}

void MQTTPublisher::configure() {
  _brokerHost = parameter("broker_host").toString();
  _brokerPort = parameter("broker_port").toInt();
  _topic = parameter("topic").toString();
  _clientId = parameter("client_id").toString();
  // Note: batch_size and batch_interval_ms parameters kept for compatibility but not used
  // All commands are published immediately upon arrival
  
  reset();
}

void MQTTPublisher::reset() {
  Algorithm::reset();
  _timeInitialized = false;

  initializeMQTT();
}

void MQTTPublisher::set_logger(::MQTTCommandLogger *logger)
{
  _commandLogger = logger;
}

void MQTTPublisher::setStartTime(time_t unix_sec, long microseconds)
{
  _startUnixTime = unix_sec;
  _startMicroseconds = microseconds;
  _startTimeSec = static_cast<Real>(unix_sec) + static_cast<Real>(microseconds) / 1000000.0;
  _timeInitialized = true;

  std::cout << "MQTTPublisher: Pipeline start time set - "
            << "Unix time: " << _startUnixTime
            << ", microseconds: " << _startMicroseconds
            << " (from chrony NTP server)" << std::endl;
}

void MQTTPublisher::initializeMQTT()
{
  try
  {
    // Create broker URI
    std::ostringstream uri;
    uri << "tcp://" << _brokerHost << ":" << _brokerPort;
    
    // Create async client
    _mqttClient = std::make_unique<mqtt::async_client>(uri.str(), _clientId);
    
    // Set connection options
    _connOpts = std::make_unique<mqtt::connect_options>();
    _connOpts->set_clean_session(true);
    _connOpts->set_automatic_reconnect(true);
    
    // Connect to broker
    std::cout << "MQTTPublisher: Connecting to broker at " << uri.str() << "..." << std::endl;
    _mqttClient->connect(*_connOpts)->wait();
    _mqttConnected = true;
    std::cout << "MQTTPublisher: Connected to MQTT broker" << std::endl;
  }
  catch (const mqtt::exception &e)
  {
    std::cerr << "MQTTPublisher: MQTT connection error: " << e.what() << std::endl;
    _mqttConnected = false;
  }
  catch (const std::exception &e)
  {
    std::cerr << "MQTTPublisher: Error initializing MQTT: " << e.what() << std::endl;
    _mqttConnected = false;
  }
}

void MQTTPublisher::cleanupMQTT() {
  if (_mqttClient && _mqttConnected) {
    try {
      _mqttClient->disconnect()->wait();
      _mqttConnected = false;
      std::cout << "MQTTPublisher: Disconnected from MQTT broker" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "MQTTPublisher: Error disconnecting: " << e.what() << std::endl;
    }
  }
  _mqttClient.reset();
  _connOpts.reset();
}

AlgorithmStatus MQTTPublisher::process() {
  AlgorithmStatus status = acquireData();
  if (status != OK) return status;

  const std::vector<std::vector<LightingCommand>>& inTokens = _in.tokens();
  
  // If no input or empty input, just release and return
  if (inTokens.empty() || inTokens[0].empty()) {
    releaseData();
    return OK;
  }
  
  // Get lighting commands directly (no parsing needed!)
  const std::vector<LightingCommand>& commands = inTokens[0];
  
  // Publish each command immediately as it arrives (no batching, no delay)
  for (const auto& cmd : commands) {
    publishSingleCommand(cmd);
  }
  
  releaseData();
  return OK;
}


void MQTTPublisher::convertToUnixTime(const LightingCommand& cmd, time_t& unixTime, long& microseconds) {
  // Time should always be initialized via setStartTime() before processing starts
  // The predictor's t_pred_sec is relative to when processing started (in seconds since start)
  // We add it to the Unix time when processing started to get absolute Unix time

  // Split cmd.tPredSec into integer seconds and fractional microseconds
  // This avoids floating point precision issues with large Unix timestamps
  Real tPredSec = cmd.tPredSec;
  
  // Get integer seconds from prediction time
  time_t predSeconds = static_cast<time_t>(std::floor(tPredSec));
  
  // Get fractional seconds and convert to microseconds with high precision
  // Use a more precise method: multiply first, then extract integer part
  Real fractionalSec = tPredSec - static_cast<Real>(predSeconds);
  
  // Convert fractional seconds to microseconds with proper rounding
  // Multiply by 1e6 and round to nearest integer
  long predMicroseconds = static_cast<long>(std::round(fractionalSec * 1000000.0));
  
  // Add start time components separately to avoid precision loss
  // Start with seconds
  unixTime = _startUnixTime + predSeconds;
  
  // Add microseconds (handling overflow)
  microseconds = _startMicroseconds + predMicroseconds;
  
  // Handle overflow: if microseconds >= 1000000, carry over to seconds
  if (microseconds >= 1000000) {
    time_t carrySeconds = microseconds / 1000000;
    unixTime += carrySeconds;
    microseconds = microseconds % 1000000;
  }
  
  // Handle underflow: if microseconds < 0, borrow from seconds
  // (shouldn't happen in normal operation, but handle for safety)
  if (microseconds < 0) {
    time_t borrowSeconds = (microseconds - 999999) / 1000000; // Negative division
    unixTime += borrowSeconds;
    microseconds = microseconds - (borrowSeconds * 1000000);
  }
  
  // Final safety check
  if (microseconds < 0 || microseconds >= 1000000) {
    // This should never happen, but if it does, normalize
    time_t adjust = microseconds / 1000000;
    unixTime += adjust;
    microseconds = microseconds - (adjust * 1000000);
    if (microseconds < 0) {
      unixTime--;
      microseconds += 1000000;
    }
  }
}

std::string MQTTPublisher::serializeMQTTMessage(time_t unixTime, long microseconds,
                                                 Real confidence, int r, int g, int b, const std::string& eventId) {
  std::ostringstream oss;
  oss << std::fixed;
  oss << "{\"unix_time\":" << unixTime
      << ",\"microseconds\":" << microseconds
      << ",\"confidence\":" << confidence
      << ",\"r\":" << r
      << ",\"g\":" << g
      << ",\"b\":" << b
      << ",\"event_id\":\"" << eventId << "\"}";
  return oss.str();
}

void MQTTPublisher::publishSingleCommand(const LightingCommand& cmd) {
  if (!_mqttConnected || !_mqttClient) {
    return;
  }
  
  try {
    // Time should already be initialized via setStartTime() before processing starts
    if (!_timeInitialized)
    {

      time_t pipelineStartUnixTime;
      long pipelineStartMicroseconds;

      if (NTPTimeClient::getTimeFromChrony(pipelineStartUnixTime, pipelineStartMicroseconds))
      {
        std::cerr << "Pipeline start time from chrony: " << pipelineStartUnixTime
                  << "." << pipelineStartMicroseconds << std::endl;
      }
      else
      {
        // Fallback to system time if chrony unavailable
        NTPTimeClient::getSystemTime(pipelineStartUnixTime, pipelineStartMicroseconds);
        std::cerr << "Pipeline start time from system (chrony unavailable): "
                  << pipelineStartUnixTime << "." << pipelineStartMicroseconds << std::endl;
      }

      // Pass start time to MQTTPublisher for accurate audio time to Unix time mapping
      // Get the predicted time (tpredsec) from LightingCommand and adjust start time accordingly
      // This makes the mapping from 'audio time' to Unix wall time accurate
      // tpredsec is expected to be in seconds (float/double)
      time_t predSeconds = static_cast<time_t>(std::floor(cmd.tPredSec));
      Real fractionalSec = cmd.tPredSec - static_cast<Real>(predSeconds);
      long predMicroseconds = static_cast<long>(std::round(fractionalSec * 1000000.0));

      _startUnixTime = pipelineStartUnixTime - predSeconds;
      _startMicroseconds = pipelineStartMicroseconds - predMicroseconds;
      if (_startMicroseconds < 0)
      {
        time_t borrowSeconds = (_startMicroseconds - 999999) / 1000000; // Negative division
        _startUnixTime += borrowSeconds;
        _startMicroseconds = _startMicroseconds - (borrowSeconds * 1000000);
      }

      setStartTime(_startUnixTime, _startMicroseconds);
    }

    time_t unixTime;
    long microseconds;
    convertToUnixTime(cmd, unixTime, microseconds);
    
    std::string payload = serializeMQTTMessage(unixTime, microseconds, cmd.confidence, cmd.r, cmd.g, cmd.b, cmd.eventId);
    
    // Create MQTT message
    mqtt::message_ptr msg = mqtt::make_message(_topic, payload);
    msg->set_qos(1); // QoS 1 for reliability
    
    // Publish (truly async, non-blocking - fire and forget)
    // Don't wait for completion to avoid blocking the pipeline
    _mqttClient->publish(msg);

    // Log command if logger is set
    if (_commandLogger)
    {
      _commandLogger->log_command(cmd);
    }

    // Reduced console output - only log occasionally to avoid blocking
    // Commented out for performance - uncomment for debugging
    // std::cout << "MQTTPublisher: Published event - ID=" << cmd.eventId 
    //           << ", RGB=(" << cmd.r << "," << cmd.g << "," << cmd.b << ")"
    //           << ", time=" << unixTime << "." << std::setfill('0') << std::setw(6) << microseconds << std::endl;
    
  } catch (const mqtt::exception& e) {
    std::cerr << "MQTTPublisher: MQTT publish error: " << e.what() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "MQTTPublisher: Error publishing command: " << e.what() << std::endl;
  }
}

// publishBatch() removed - all commands are now published immediately

} // namespace streaming
} // namespace essentia

