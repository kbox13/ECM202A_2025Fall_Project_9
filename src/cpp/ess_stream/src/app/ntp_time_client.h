#ifndef NTP_TIME_CLIENT_H
#define NTP_TIME_CLIENT_H

#include <time.h>
#include <sys/time.h>
#include <cstdint>

/**
 * NTPTimeClient - Query local chrony NTP server for precise Unix time
 * 
 * This class provides methods to get Unix time from the local chrony NTP server
 * running on localhost:123. Falls back to system time if NTP query fails.
 */
class NTPTimeClient {
public:
    /**
     * Get Unix time from local chrony NTP server
     * 
     * @param unix_sec Output parameter for Unix seconds
     * @param microseconds Output parameter for microseconds
     * @return true on success, false on failure (will use system time fallback)
     */
    static bool getTimeFromChrony(time_t& unix_sec, long& microseconds);
    
    /**
     * Fallback: Get time from system (if NTP fails)
     * 
     * @param unix_sec Output parameter for Unix seconds
     * @param microseconds Output parameter for microseconds
     */
    static void getSystemTime(time_t& unix_sec, long& microseconds);
    
private:
    // NTP epoch is Jan 1, 1900, Unix epoch is Jan 1, 1970
    // Difference is 2208988800 seconds
    static constexpr time_t NTP_EPOCH_OFFSET = 2208988800;
    
    /**
     * Convert NTP timestamp (64-bit: 32-bit seconds + 32-bit fraction) to Unix time
     * 
     * @param ntp_seconds NTP seconds (since 1900)
     * @param ntp_fraction NTP fraction (0-2^32-1, represents fractional seconds)
     * @param unix_sec Output Unix seconds
     * @param microseconds Output microseconds
     */
    static void ntpToUnixTime(uint32_t ntp_seconds, uint32_t ntp_fraction, 
                             time_t& unix_sec, long& microseconds);
};

#endif // NTP_TIME_CLIENT_H

