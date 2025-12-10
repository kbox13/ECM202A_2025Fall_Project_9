#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <map>
#include <algorithm>
#include <numeric>
#include <thread>
#include <ctime>

#include <mqtt/async_client.h>
#include <mqtt/connect_options.h>
#include <mqtt/message.h>
#include "ntp_time_client.h"

// MQTT Topics
const char* TOPIC_LATENCY_REQUEST = "beat/test/latency_request";
const char* TOPIC_LATENCY_RESPONSE = "beat/test/latency_response";
const char* TOPIC_TIME_SYNC_REQUEST = "beat/test/time_sync_request";
const char* TOPIC_TIME_SYNC_RESPONSE = "beat/test/time_sync_response";

// Statistics helper class
class Statistics {
public:
    void add(double value) {
        values.push_back(value);
    }

    void clear() {
        values.clear();
    }

    size_t count() const {
        return values.size();
    }

    double min() const {
        if (values.empty()) return 0.0;
        return *std::min_element(values.begin(), values.end());
    }

    double max() const {
        if (values.empty()) return 0.0;
        return *std::max_element(values.begin(), values.end());
    }

    double mean() const {
        if (values.empty()) return 0.0;
        double sum = std::accumulate(values.begin(), values.end(), 0.0);
        return sum / values.size();
    }

    double stddev() const {
        if (values.size() < 2) return 0.0;
        double m = mean();
        double sum_sq_diff = 0.0;
        for (double v : values) {
            double diff = v - m;
            sum_sq_diff += diff * diff;
        }
        return std::sqrt(sum_sq_diff / (values.size() - 1));
    }

private:
    std::vector<double> values;
};

// Latency tester
class LatencyTester {
public:
    LatencyTester(mqtt::async_client& client) : mqttClient(client) {
        // Subscribe to response topic
        mqttClient.subscribe(TOPIC_LATENCY_RESPONSE, 1)->wait();
    }

    void runTest(int numSamples) {
        numSamples = numSamples > 100 ? 100 : numSamples;
        std::cout << "Running latency test (" << numSamples << " samples)..." << std::endl;
        
        stats.clear();
        pendingRequests.clear();

        for (int i = 0; i < numSamples; ++i) {
            // Get high-precision timestamp
            auto sendTime = std::chrono::high_resolution_clock::now();
            auto sendTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                sendTime.time_since_epoch()).count();
            auto sendTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(
                sendTime.time_since_epoch()).count();

            // Create request message
            std::ostringstream oss;
            oss << "{\"request_id\":" << i 
                << ",\"host_timestamp_ms\":" << sendTimeMs
                << ",\"host_timestamp_us\":" << sendTimeUs << "}";
            
            std::string payload = oss.str();
            mqtt::message_ptr msg = mqtt::make_message(TOPIC_LATENCY_REQUEST, payload);
            msg->set_qos(1);

            // Store request time
            pendingRequests[i] = sendTime;

            // Publish request
            mqttClient.publish(msg)->wait();

            // Wait for response (with timeout)
            auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            bool received = false;
            while (std::chrono::steady_clock::now() < timeout && !received) {
                // Check if we have a response for this request
                if (receivedResponses.find(i) != receivedResponses.end()) {
                    auto recvTime = receivedResponses[i];
                    auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
                        recvTime - sendTime).count() / 1000.0; // Convert to ms
                    stats.add(latency);
                    receivedResponses.erase(i);
                    received = true;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }

            if (!received) {
                std::cerr << "Warning: No response for request " << i << std::endl;
            }

            // Small delay between requests
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        std::cout << "Latency test complete. Received " << stats.count() << " responses." << std::endl;
    }

    void handleResponse(const std::string& payload) {
        // Parse JSON response
        // Format: {"request_id":N,"arduino_timestamp_ms":M,"arduino_timestamp_us":U,...}
        try {
            size_t reqIdPos = payload.find("\"request_id\"");
            if (reqIdPos == std::string::npos) return;

            // Extract request_id
            size_t colonPos = payload.find(':', reqIdPos);
            if (colonPos == std::string::npos) return;
            
            size_t commaPos = payload.find_first_of(",}", colonPos);
            if (commaPos == std::string::npos) return;
            
            std::string reqIdStr = payload.substr(colonPos + 1, commaPos - colonPos - 1);
            // Remove whitespace
            reqIdStr.erase(0, reqIdStr.find_first_not_of(" \t"));
            reqIdStr.erase(reqIdStr.find_last_not_of(" \t") + 1);
            
            int requestId = std::stoi(reqIdStr);

            // Record response time
            auto recvTime = std::chrono::high_resolution_clock::now();
            receivedResponses[requestId] = recvTime;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing latency response: " << e.what() << std::endl;
        }
    }

    Statistics getStats() const {
        return stats;
    }

private:
    mqtt::async_client& mqttClient;
    Statistics stats;
    std::map<int, std::chrono::high_resolution_clock::time_point> pendingRequests;
    std::map<int, std::chrono::high_resolution_clock::time_point> receivedResponses;
};

// Time sync tester
class TimeSyncTester {
public:
    TimeSyncTester(mqtt::async_client& client) : mqttClient(client) {
        // Subscribe to response topic
        mqttClient.subscribe(TOPIC_TIME_SYNC_RESPONSE, 1)->wait();
    }

    void runTest(int numSamples) {
        std::cout << "Running time sync test (" << numSamples << " samples)..." << std::endl;
        std::cout << "Note: Arduino must be time-synced (via NTP) for accurate results." << std::endl;
        std::cout << "      Using NTP-style symmetric measurement to account for network delay." << std::endl;
        
        timeOffsets.clear();
        networkDelays.clear();
        pendingRequests.clear();

        for (int i = 0; i < numSamples; ++i) {
            // Get host time from NTP (preferred) or system time
            time_t hostUnixSec;
            long hostMicroseconds;
            bool ntpSuccess = NTPTimeClient::getTimeFromChrony(hostUnixSec, hostMicroseconds);
            if (!ntpSuccess) {
                NTPTimeClient::getSystemTime(hostUnixSec, hostMicroseconds);
                std::cout << "Warning: Failed to get time from chrony NTP server. Using system time." << std::endl;
            }

            // Capture T1: Host send time (high precision)
            auto t1_chrono = std::chrono::high_resolution_clock::now();
            long long t1_totalMicros = static_cast<long long>(hostUnixSec) * 1000000LL + hostMicroseconds;

            // Create request message
            std::ostringstream oss;
            oss << "{\"request_id\":" << i 
                << ",\"host_unix_time\":" << hostUnixSec
                << ",\"host_microseconds\":" << hostMicroseconds << "}";
            
            std::string payload = oss.str();
            mqtt::message_ptr msg = mqtt::make_message(TOPIC_TIME_SYNC_REQUEST, payload);
            msg->set_qos(1);

            // Store T1 timestamp (both Unix time and chrono for precision)
            RequestTimestamp reqTimestamp;
            reqTimestamp.unix_sec = hostUnixSec;
            reqTimestamp.unix_usec = hostMicroseconds;
            reqTimestamp.chrono_time = t1_chrono;
            pendingRequests[i] = reqTimestamp;

            // Publish request
            mqttClient.publish(msg)->wait();

            // Wait for response (with timeout)
            auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            bool received = false;
            while (std::chrono::steady_clock::now() < timeout && !received) {
                if (receivedResponses.find(i) != receivedResponses.end()) {
                    auto responseData = receivedResponses[i];
                    auto requestData = pendingRequests[i];
                    
                    // Extract timestamps:
                    // T1: Host send time (from request)
                    // T2: Arduino receive time (approximated as Arduino send time, since processing is fast)
                    // T3: Arduino send time (from response)
                    // T4: Host receive time (captured in handleResponse)
                    
                    time_t t1_sec = requestData.unix_sec;
                    long t1_usec = requestData.unix_usec;
                    auto t1_chrono = requestData.chrono_time;
                    
                    time_t t2_sec = responseData.arduino_unix_sec;  // Approximate: Arduino receive ≈ send
                    long t2_usec = responseData.arduino_microseconds;
                    
                    time_t t3_sec = responseData.arduino_unix_sec;  // Arduino send time
                    long t3_usec = responseData.arduino_microseconds;
                    
                    auto t4_chrono = responseData.host_receive_time;
                    
                    // Check if Arduino time is valid
                    bool arduinoTimeValid = (t2_sec > 1000000000 && t2_sec < 5000000000);
                    if (!arduinoTimeValid) {
                        std::cerr << "Warning: Arduino time invalid. Skipping sample " << i << std::endl;
                        receivedResponses.erase(i);
                        received = true;
                        continue;
                    }
                    
                    // Convert T4 (host receive) to Unix time using T1 as reference
                    // Calculate elapsed time between T1 and T4 using chrono
                    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                        t4_chrono - t1_chrono).count();
                    long long t4_totalMicros = (static_cast<long long>(t1_sec) * 1000000LL + t1_usec) + elapsed;
                    time_t t4_sec = t4_totalMicros / 1000000LL;
                    long t4_usec = t4_totalMicros % 1000000LL;
                    
                    // Convert all to total microseconds for calculation
                    long long t1_total = static_cast<long long>(t1_sec) * 1000000LL + t1_usec;
                    long long t2_total = static_cast<long long>(t2_sec) * 1000000LL + t2_usec;
                    long long t3_total = static_cast<long long>(t3_sec) * 1000000LL + t3_usec;
                    long long t4_total = static_cast<long long>(t4_sec) * 1000000LL + t4_usec;
                    
                    // Calculate network delay: delay = (T4 - T1) - (T3 - T2)
                    // Since T2 ≈ T3 (Arduino processing is fast), delay ≈ (T4 - T1)
                    long long delay_us = elapsed;
                    double delayMs = delay_us / 1000.0;
                    
                    // Calculate NTP-style offset: offset = ((T2 - T1) + (T3 - T4)) / 2
                    // This cancels out symmetric network delay
                    long long offset_us = ((t2_total - t1_total) + (t3_total - t4_total)) / 2;
                    offset_us = std::abs(offset_us);
                    // subtract the rounded down half of the roundtrip delay from the offset
                    // this approximates how close the clocks actually are, removed the 
                    double offsetMs = offset_us / 1000.0;
                    // std::cout << "Delay sample " << i << ": " << delayMs << " ms" << std::endl;
                    // std::cout << "Offset sample " << i << ": " << offsetMs << " ms" << std::endl;
                    
                    
                    // Reject if offset is unreasonably large (> 1 hour)
                    if (std::abs(offsetMs) > 3600000.0) {
                        std::cerr << "Warning: Offset too large (" << offsetMs 
                                  << " ms). Skipping sample " << i << std::endl;
                        receivedResponses.erase(i);
                        received = true;
                        continue;
                    }

                    // Filter out measurements with excessive delay (asymmetric network)
                    // Include only if delay < 20ms (indicates network problems)
                    if (delayMs < 20.0) {
                        timeOffsets.push_back(offsetMs);
                        continue;
                    }
                    
                    // Valid measurement
                    networkDelays.push_back(delayMs);
                    receivedResponses.erase(i);
                    received = true;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }

            if (!received) {
                std::cerr << "Warning: No response for time sync request " << i << std::endl;
            }

            // Small delay between requests
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        std::cout << "Time sync test complete. Received " << timeOffsets.size() << " valid responses." << std::endl;
        if (timeOffsets.empty()) {
            std::cerr << "WARNING: No valid time sync measurements! Arduino may not be time-synced." << std::endl;
            std::cerr << "         Check Arduino serial output for NTP sync status." << std::endl;
        }
    }

    void handleResponse(const std::string& payload) {
        // Capture T4: Host receive time immediately when message arrives
        auto t4_receive_time = std::chrono::high_resolution_clock::now();
        
        // Parse JSON response
        // Format: {"request_id":N,"arduino_unix_time":M,"arduino_microseconds":U}
        // Debug: print first few responses to see what Arduino is reporting
        static int debugCount = 0;
        if (debugCount < 3) {
            std::cout << "DEBUG: Arduino response #" << debugCount << ": " << payload << std::endl;
            debugCount++;
        }
        
        try {
            size_t reqIdPos = payload.find("\"request_id\"");
            if (reqIdPos == std::string::npos) return;

            // Extract request_id
            size_t colonPos = payload.find(':', reqIdPos);
            if (colonPos == std::string::npos) return;
            
            size_t commaPos = payload.find_first_of(",}", colonPos);
            if (commaPos == std::string::npos) return;
            
            std::string reqIdStr = payload.substr(colonPos + 1, commaPos - colonPos - 1);
            reqIdStr.erase(0, reqIdStr.find_first_not_of(" \t"));
            reqIdStr.erase(reqIdStr.find_last_not_of(" \t") + 1);
            int requestId = std::stoi(reqIdStr);

            // Extract arduino_unix_time
            size_t unixPos = payload.find("\"arduino_unix_time\"");
            if (unixPos == std::string::npos) return;
            size_t unixColonPos = payload.find(':', unixPos);
            if (unixColonPos == std::string::npos) return;
            size_t unixCommaPos = payload.find_first_of(",}", unixColonPos);
            if (unixCommaPos == std::string::npos) return;
            std::string unixStr = payload.substr(unixColonPos + 1, unixCommaPos - unixColonPos - 1);
            unixStr.erase(0, unixStr.find_first_not_of(" \t"));
            unixStr.erase(unixStr.find_last_not_of(" \t") + 1);
            time_t arduinoUnixSec = std::stol(unixStr);

            // Extract arduino_microseconds
            size_t microsPos = payload.find("\"arduino_microseconds\"");
            if (microsPos == std::string::npos) return;
            size_t microsColonPos = payload.find(':', microsPos);
            if (microsColonPos == std::string::npos) return;
            size_t microsEndPos = payload.find_first_of(",}", microsColonPos);
            if (microsEndPos == std::string::npos) return;
            std::string microsStr = payload.substr(microsColonPos + 1, microsEndPos - microsColonPos - 1);
            microsStr.erase(0, microsStr.find_first_not_of(" \t"));
            microsStr.erase(microsStr.find_last_not_of(" \t") + 1);
            long arduinoMicroseconds = std::stol(microsStr);

            // Store response with T4 timestamp
            ResponseData responseData;
            responseData.arduino_unix_sec = arduinoUnixSec;
            responseData.arduino_microseconds = arduinoMicroseconds;
            responseData.host_receive_time = t4_receive_time;
            receivedResponses[requestId] = responseData;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing time sync response: " << e.what() << std::endl;
        }
    }

    Statistics getOffsetStats() const {
        Statistics stats;
        for (double offset : timeOffsets) {
            stats.add(offset);
        }
        return stats;
    }

    Statistics getDelayStats() const {
        Statistics stats;
        for (double delay : networkDelays) {
            stats.add(delay);
        }
        return stats;
    }

private:
    // Request timestamp structure (T1)
    struct RequestTimestamp {
        time_t unix_sec;
        long unix_usec;
        std::chrono::high_resolution_clock::time_point chrono_time;
    };
    
    // Response data structure (T2, T3, T4)
    struct ResponseData {
        time_t arduino_unix_sec;      // T2/T3 (Arduino receive/send, approximated as same)
        long arduino_microseconds;
        std::chrono::high_resolution_clock::time_point host_receive_time;  // T4
    };
    
    mqtt::async_client& mqttClient;
    std::vector<double> timeOffsets;
    std::vector<double> networkDelays;
    std::map<int, RequestTimestamp> pendingRequests;
    std::map<int, ResponseData> receivedResponses;
};

// MQTT callback handler
class TestCallback : public virtual mqtt::callback {
public:
    TestCallback(LatencyTester& latencyTester, TimeSyncTester& timeSyncTester)
        : latencyTester(latencyTester), timeSyncTester(timeSyncTester) {}

    void message_arrived(mqtt::const_message_ptr msg) override {
        std::string topic = msg->get_topic();
        std::string payload = msg->to_string();

        if (topic == TOPIC_LATENCY_RESPONSE) {
            latencyTester.handleResponse(payload);
        } else if (topic == TOPIC_TIME_SYNC_RESPONSE) {
            timeSyncTester.handleResponse(payload);
        }
    }

    void connection_lost(const std::string& cause) override {
        std::cerr << "Connection lost: " << cause << std::endl;
    }

    void delivery_complete(mqtt::delivery_token_ptr tok) override {
        // Not needed for this test
    }

private:
    LatencyTester& latencyTester;
    TimeSyncTester& timeSyncTester;
};

void printResults(const Statistics& latencyStats, const Statistics& timeSyncStats, const Statistics& delayStats) {
    std::cout << "\n=== Test Results ===" << std::endl;
    
    std::cout << "\nNetwork Latency (MQTT RTT):" << std::endl;
    std::cout << "  Samples:  " << latencyStats.count() << std::endl;
    if (latencyStats.count() > 0) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Min:     " << latencyStats.min() << " ms" << std::endl;
        std::cout << "  Max:     " << latencyStats.max() << " ms" << std::endl;
        std::cout << "  Mean:    " << latencyStats.mean() << " ms" << std::endl;
        std::cout << "  StdDev:  " << latencyStats.stddev() << " ms" << std::endl;
    }

    std::cout << "\nNetwork Delay (One-way, calculated):" << std::endl;
    std::cout << "  Samples:  " << delayStats.count() << std::endl;
    if (delayStats.count() > 0) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Min:     " << delayStats.min() << " ms" << std::endl;
        std::cout << "  Max:     " << delayStats.max() << " ms" << std::endl;
        std::cout << "  Mean:    " << delayStats.mean() << " ms" << std::endl;
        std::cout << "  StdDev:  " << delayStats.stddev() << " ms" << std::endl;
    }

    std::cout << "\nTime Sync Difference (NTP-style, delay-compensated):" << std::endl;
    std::cout << "  Samples:  " << timeSyncStats.count() << std::endl;
    if (timeSyncStats.count() > 0) {
        std::cout << std::fixed << std::setprecision(3);
        double meanOffset = timeSyncStats.mean();
        
        // Check for obviously wrong values (more than 1 hour difference suggests unsynced)
        if (std::abs(meanOffset) > 3600000.0) {
            std::cout << "  WARNING: Mean offset is very large (" << meanOffset << " ms)" << std::endl;
            std::cout << "           Arduino may not be properly time-synced!" << std::endl;
        }
        
        std::cout << "  Mean Offset: " << meanOffset << " ms";
        if (meanOffset > 10.0) {
            std::cout << " (Arduino ahead)" << std::endl;
        } else if (meanOffset < -10.0) {
            std::cout << " (Host ahead)" << std::endl;
        } else {
            std::cout << " (Well synchronized)" << std::endl;
        }
        std::cout << "  Min:     " << timeSyncStats.min() << " ms" << std::endl;
        std::cout << "  Max:     " << timeSyncStats.max() << " ms" << std::endl;
        std::cout << "  StdDev:  " << timeSyncStats.stddev() << " ms" << std::endl;
    } else {
        std::cout << "  No valid measurements (Arduino not time-synced)" << std::endl;
    }
}

int main(int argc, char** argv) {
    // Parse command line arguments
    std::string broker = "localhost";
    int port = 1883;
    int numSamples = 100;

    if (argc > 1) {
        broker = argv[1];
    }
    if (argc > 2) {
        port = std::stoi(argv[2]);
    }
    if (argc > 3) {
        numSamples = std::stoi(argv[3]);
    }

    std::cout << "=== Arduino Latency and Time Sync Test ===" << std::endl;
    std::cout << "Broker: " << broker << ":" << port << std::endl;
    std::cout << "Samples: " << numSamples << std::endl;
    std::cout << std::endl;

    try {
        // Create MQTT client
        std::ostringstream uri;
        uri << "tcp://" << broker << ":" << port;
        std::string clientId = "latency_test_client_" + std::to_string(std::time(nullptr));
        
        mqtt::async_client client(uri.str(), clientId);
        
        // Set connection options
        mqtt::connect_options connOpts;
        connOpts.set_clean_session(true);
        connOpts.set_automatic_reconnect(true);

        // Connect to broker
        std::cout << "Connecting to MQTT broker..." << std::endl;
        client.connect(connOpts)->wait();
        std::cout << "Connected." << std::endl;

        // Create testers (they will subscribe to response topics)
        LatencyTester latencyTester(client);
        TimeSyncTester timeSyncTester(client);

        // Set up callback (must be after testers are created)
        TestCallback callback(latencyTester, timeSyncTester);
        client.set_callback(callback);

        // Wait a moment for subscriptions to be processed
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Run latency test
        latencyTester.runTest(numSamples);

        // Small delay between tests
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Run time sync test
        timeSyncTester.runTest(numSamples);

        // Get results
        Statistics latencyStats = latencyTester.getStats();
        Statistics timeSyncStats = timeSyncTester.getOffsetStats();
        Statistics delayStats = timeSyncTester.getDelayStats();

        // Print results
        printResults(latencyStats, timeSyncStats, delayStats);

        // Disconnect
        client.disconnect()->wait();
        std::cout << "\nDisconnected." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

