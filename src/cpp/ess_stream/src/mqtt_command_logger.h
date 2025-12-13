#ifndef MQTT_COMMAND_LOGGER_H
#define MQTT_COMMAND_LOGGER_H

/*
 * MQTTCommandLogger
 * - Accumulates tPredSec values per instrument during runtime
 * - Writes one JSON object per instrument at shutdown
 * - Thread-safe accumulation with mutex
 */

#include <essentia/types.h>
#include <fstream>
#include <string>
#include <mutex>
#include <map>
#include <vector>
#include <chrono>
#include <iomanip>
#include <ctime>

// Forward declaration of LightingCommand (defined in prediction_types.h)
namespace essentia {
namespace streaming {
struct LightingCommand;
}
}

class MQTTCommandLogger {
public:
    /**
     * Constructor
     * @param logDir Directory for log files (created if doesn't exist)
     */
    MQTTCommandLogger(const std::string& logDir = "logs");
    
    /**
     * Destructor - writes all accumulated data to file (final write)
     */
    ~MQTTCommandLogger();
    
    /**
     * Log a command by accumulating its tPredSec value per instrument
     * Writes to file immediately after accumulating to ensure data is saved incrementally
     * @param cmd LightingCommand containing instrument and tPredSec
     */
    void log_command(const essentia::streaming::LightingCommand& cmd);
    
    /**
     * Check if logger is enabled
     */
    bool is_enabled() const { return true; }  // Always enabled (writes at shutdown)
    
    /**
     * Get the log filename (for debugging/info)
     */
    const std::string& get_log_filename() const { return _log_filename; }

private:
    std::string _log_filename;
    std::string _log_dir;
    std::mutex _accumulate_mutex;  // Thread-safe accumulation
    std::map<std::string, std::vector<essentia::Real>> _accumulated_data;  // instrument -> vector of tPredSec
    
    /**
     * Generate timestamped log filename
     */
    std::string get_timestamped_filename(const std::string& logDir);
    
    /**
     * Helper to ensure log directory exists (platform-specific)
     */
    void ensure_log_directory(const std::string& logDir);
    
    /**
     * Write all accumulated data to file as JSON Lines
     * Rewrites the entire file to maintain the same structure (one object per instrument)
     */
    void write_to_file();
};

#endif // MQTT_COMMAND_LOGGER_H

