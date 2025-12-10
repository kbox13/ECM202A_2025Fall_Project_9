#include "mqtt_command_logger.h"
#include "prediction_types.h"  // For LightingCommand definition
#include <sstream>
#include <iostream>
#include <sys/stat.h>  // For mkdir on Unix/macOS
#include <sys/types.h>
#include <cerrno>  // For errno

// Platform-specific directory creation
#ifdef _WIN32
#include <direct.h>
#define mkdir(dir, mode) _mkdir(dir)
#endif

MQTTCommandLogger::MQTTCommandLogger(const std::string& logDir)
    : _log_dir(logDir) {
    
    // Create log directory if it doesn't exist
    ensure_log_directory(logDir);
    
    // Get timestamped filename
    _log_filename = get_timestamped_filename(logDir);
    
    // Write initial empty file with header
    write_to_file();
    
    std::cerr << "MQTTCommandLogger: Writing to " << _log_filename << " (incremental updates)" << std::endl;
}

MQTTCommandLogger::~MQTTCommandLogger() {
    // Final write to ensure all data is saved (file should already be up to date from incremental writes)
    write_to_file();
    std::cerr << "MQTTCommandLogger: Final write completed for " << _log_filename << std::endl;
}

void MQTTCommandLogger::log_command(const essentia::streaming::LightingCommand& cmd) {
    // Thread-safely accumulate tPredSec per instrument
    {
        std::lock_guard<std::mutex> lock(_accumulate_mutex);
        
        // Append tPredSec to the appropriate instrument's vector
        _accumulated_data[cmd.instrument].push_back(cmd.tPredSec);
    }
    
    // Write to file immediately after accumulating to ensure incremental saves
    // This rewrites the entire file but maintains the same structure
    write_to_file();
}

void MQTTCommandLogger::write_to_file() {
    std::ofstream log_file;
    log_file.open(_log_filename, std::ios::out | std::ios::trunc);  // Truncate to rewrite entire file
    
    if (!log_file.is_open()) {
        std::cerr << "MQTTCommandLogger: Warning - Failed to open log file: " << _log_filename << std::endl;
        return;
    }
    
    // Write header
    log_file << "# MQTT Command Log\n";
    log_file << "# Format: JSON Lines (one object per instrument)\n";
    log_file << "# Each line contains instrument name and all tPredSec values for the entire run\n";
    log_file << "# Updated incrementally as commands are logged\n";
    log_file << "#\n";
    
    // Define all 5 instruments in order
    static const char* inst_names[] = {"kick", "snare", "clap", "chat", "ohc"};
    
    // Get all accumulated data (thread-safe access)
    std::map<std::string, std::vector<essentia::Real>> data_copy;
    {
        std::lock_guard<std::mutex> lock(_accumulate_mutex);
        data_copy = _accumulated_data;
    }
    
    // Write one JSON object per instrument
    for (int i = 0; i < 5; ++i) {
        const std::string& inst = inst_names[i];
        
        // Get accumulated data for this instrument
        auto it = data_copy.find(inst);
        const std::vector<essentia::Real>& tPredSec_values = (it != data_copy.end()) ? it->second : std::vector<essentia::Real>();
        
        // Write JSON object
        log_file << std::fixed << std::setprecision(6);
        log_file << "{\"instrument\":\"" << inst << "\",\"tPredSec\":[";
        
        // Write tPredSec array
        for (size_t j = 0; j < tPredSec_values.size(); ++j) {
            if (j > 0) {
                log_file << ",";
            }
            log_file << tPredSec_values[j];
        }
        
        log_file << "]}\n";
    }
    
    log_file.flush();  // Ensure data is written immediately
    log_file.close();
}

std::string MQTTCommandLogger::get_timestamped_filename(const std::string& logDir) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
#ifdef _WIN32
    struct tm tm_buf;
    localtime_s(&tm_buf, &time_t);
#else
    struct tm tm_buf;
    localtime_r(&time_t, &tm_buf);
#endif
    
    std::ostringstream oss;
    oss << logDir << "/mqtt_commands_"
        << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << ".log";
    return oss.str();
}

void MQTTCommandLogger::ensure_log_directory(const std::string& logDir) {
    // Try to create directory (mkdir returns 0 on success, -1 if exists or error)
    int result = mkdir(logDir.c_str(), 0755);
    if (result != 0) {
        // Check if directory already exists (errno == EEXIST on Unix)
#ifdef _WIN32
        // On Windows, check GetLastError or just ignore
#else
        if (errno != EEXIST) {
            std::cerr << "MQTTCommandLogger: Warning - Could not create log directory: " 
                      << logDir << std::endl;
        }
#endif
    }
}

