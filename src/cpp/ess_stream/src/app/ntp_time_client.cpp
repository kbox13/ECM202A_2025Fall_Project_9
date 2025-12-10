#include "ntp_time_client.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <cerrno>
#include <cstdint>

// NTP packet structure (48 bytes)
struct NTPPacket {
    uint8_t li_vn_mode;      // Leap indicator (2 bits), Version (3 bits), Mode (3 bits)
    uint8_t stratum;         // Stratum level
    uint8_t poll;            // Poll interval
    int8_t precision;        // Precision
    uint32_t root_delay;     // Root delay
    uint32_t root_dispersion; // Root dispersion
    uint32_t ref_id;         // Reference ID
    uint32_t ref_ts_sec;     // Reference timestamp (seconds)
    uint32_t ref_ts_frac;    // Reference timestamp (fraction)
    uint32_t orig_ts_sec;    // Origin timestamp (seconds)
    uint32_t orig_ts_frac;   // Origin timestamp (fraction)
    uint32_t recv_ts_sec;   // Receive timestamp (seconds)
    uint32_t recv_ts_frac;   // Receive timestamp (fraction)
    uint32_t trans_ts_sec;  // Transmit timestamp (seconds)
    uint32_t trans_ts_frac;  // Transmit timestamp (fraction)
} __attribute__((packed));

bool NTPTimeClient::getTimeFromChrony(time_t& unix_sec, long& microseconds) {
    // Initialize output parameters
    unix_sec = 0;
    microseconds = 0;
    
    // Create UDP socket
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        std::cerr << "NTPTimeClient: Failed to create socket: " << strerror(errno) << std::endl;
        return false;
    }
    
    // Set socket timeout (1 second)
    struct timeval timeout;
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
        std::cerr << "NTPTimeClient: Failed to set socket timeout: " << strerror(errno) << std::endl;
        close(sockfd);
        return false;
    }
    
    // Set up server address (localhost:123)
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(123); // NTP port
    if (inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr) <= 0) {
        std::cerr << "NTPTimeClient: Invalid address" << std::endl;
        close(sockfd);
        return false;
    }
    
    // Prepare NTP request packet
    NTPPacket request;
    memset(&request, 0, sizeof(request));
    // Set mode to 3 (client) and version to 4
    // Format: LI (2 bits) | VN (3 bits) | Mode (3 bits)
    // LI=0 (no warning), VN=4 (NTP version 4), Mode=3 (client)
    request.li_vn_mode = (0 << 6) | (4 << 3) | 3;
    
    // Send request
    if (sendto(sockfd, &request, sizeof(request), 0, 
               (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "NTPTimeClient: Failed to send request: " << strerror(errno) << std::endl;
        close(sockfd);
        return false;
    }
    
    // Receive response
    NTPPacket response;
    memset(&response, 0, sizeof(response)); // Initialize to avoid garbage
    struct sockaddr_in from_addr;
    socklen_t from_len = sizeof(from_addr);
    
    ssize_t recv_len = recvfrom(sockfd, &response, sizeof(response), 0,
                                (struct sockaddr*)&from_addr, &from_len);
    
    close(sockfd);
    
    if (recv_len < sizeof(NTPPacket)) {
        std::cerr << "NTPTimeClient: Invalid response size: " << recv_len 
                  << " (expected " << sizeof(NTPPacket) << ")" << std::endl;
        return false;
    }
    
    // Validate NTP response
    uint8_t mode = response.li_vn_mode & 0x07; // Bottom 3 bits
    uint8_t version = (response.li_vn_mode >> 3) & 0x07; // Middle 3 bits
    uint8_t stratum = response.stratum;
    
    std::cout << "NTPTimeClient: Response - mode=" << (int)mode 
              << ", version=" << (int)version 
              << ", stratum=" << (int)stratum << std::endl;
    
    // Mode should be 4 (server) for a valid response
    if (mode != 4) {
        std::cerr << "NTPTimeClient: Invalid response mode: " << (int)mode 
                  << " (expected 4)" << std::endl;
        return false;
    }
    
    // Stratum should be 1-15 for a valid time source
    if (stratum == 0 || stratum > 15) {
        std::cerr << "NTPTimeClient: Invalid stratum: " << (int)stratum << std::endl;
        return false;
    }
    
    // Extract transmit timestamp (most accurate)
    uint32_t ntp_seconds = ntohl(response.trans_ts_sec);
    uint32_t ntp_fraction = ntohl(response.trans_ts_frac);
    
    std::cout << "NTPTimeClient: NTP timestamp - seconds=" << ntp_seconds 
              << ", fraction=" << ntp_fraction << std::endl;
    
    // Validate NTP timestamp is not zero (which would indicate invalid/uninitialized)
    // Valid NTP seconds should be > NTP_EPOCH_OFFSET (2208988800) for current times
    // This corresponds to Unix time > 0 (year 1970)
    if (ntp_seconds == 0) {
        std::cerr << "NTPTimeClient: Invalid NTP timestamp (zero seconds)" << std::endl;
        return false;
    }
    
    // Convert NTP time to Unix time
    ntpToUnixTime(ntp_seconds, ntp_fraction, unix_sec, microseconds);
    
    std::cout << "NTPTimeClient: Converted to Unix time - " << unix_sec 
              << "." << microseconds << std::endl;
    
    // Validate converted Unix time is reasonable (between 2001 and 2128)
    // This catches conversion errors or corrupted data
    if (unix_sec < 1000000000 || unix_sec > 5000000000) {
        std::cerr << "NTPTimeClient: Converted Unix time out of reasonable range: " 
                  << unix_sec << " (expected 1000000000-5000000000)" << std::endl;
        unix_sec = 0;
        microseconds = 0;
        return false;
    }
    
    return true;
}

void NTPTimeClient::getSystemTime(time_t& unix_sec, long& microseconds) {
    struct timeval tv;
    if (gettimeofday(&tv, nullptr) == 0) {
        unix_sec = tv.tv_sec;
        microseconds = tv.tv_usec;
        std::cout << "NTPTimeClient: Using system time - " << unix_sec 
                  << "." << microseconds << std::endl;
    } else {
        // Fallback to time() if gettimeofday fails
        unix_sec = time(nullptr);
        microseconds = 0;
        std::cerr << "NTPTimeClient: gettimeofday failed, using time() fallback: " 
                  << unix_sec << std::endl;
    }
}

void NTPTimeClient::ntpToUnixTime(uint32_t ntp_seconds, uint32_t ntp_fraction,
                                   time_t& unix_sec, long& microseconds) {
    // Convert NTP seconds (since 1900) to Unix seconds (since 1970)
    unix_sec = static_cast<time_t>(ntp_seconds) - NTP_EPOCH_OFFSET;
    
    // Convert NTP fraction (0-2^32-1) to microseconds
    // NTP fraction represents fractional seconds as: fraction / 2^32
    // So microseconds = (fraction * 1000000) / 2^32
    // Use 64-bit arithmetic to avoid overflow
    uint64_t fraction_64 = static_cast<uint64_t>(ntp_fraction);
    uint64_t micros_64 = (fraction_64 * 1000000ULL) / 4294967296ULL; // 2^32 = 4294967296
    microseconds = static_cast<long>(micros_64);
}

