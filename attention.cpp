#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>
#include<sstream>
#include <unistd.h>
#include "./include/helpers.h"
#include <sys/socket.h>
#include"server_config.h"
using namespace std;

ssize_t recv_all(int socket_fd, void* data, size_t length) {
    char* buffer = static_cast<char*>(data);
    size_t total = 0;

    while (total < length) {
        ssize_t r = recv(socket_fd, buffer + total, length - total, 0);
        
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }

        if (r == 0) return 0;

        total += r;
    }

    return total;
}

bool send_all(int socket_fd, const void* data, size_t length) {
    const char* buf = (const char*)data;
    size_t sent_total = 0;

    while (sent_total < length) {
        ssize_t s = send(socket_fd, buf + sent_total, length - sent_total, 0);

        if (s < 0) {
            if (errno == EINTR) continue;
            return false;
        }

        if (s == 0) return false;

        sent_total += s;
    }

    return true;
}

bool send_with_size(int socket_fd, const void* data, uint32_t length) {
    uint32_t net = htonl(length);

    if (!send_all(socket_fd, &net, sizeof(net)))
        return false;

    return send_all(socket_fd, data, length);
}

bool recv_with_size(int socket_fd, string& out) {
    uint32_t net = 0;

    if (recv_all(socket_fd, &net, sizeof(net)) <= 0)
        return false;

    uint32_t len = ntohl(net);
    out.resize(len);

    return recv_all(socket_fd, out.data(), len) > 0;
}
void show_pair_result(string& response, int& i){
    uint64_t size;
    memcpy(&size, response.data() + i, sizeof(uint64_t));
    i += sizeof(uint64_t);
    for(uint64_t j = 0;j < size && i < response.size();j++){
        uint64_t value;
        memcpy(&value, response.data() + i, sizeof(uint64_t));
        i += sizeof(uint64_t);
        double val;
        memcpy(&val, response.data() + i, sizeof(double));
        i += sizeof(double);
        cout << "(" << value << " , " << val << ")\n";
        
    }
}
string deserialize(const string& str, int& i, const char del){
    size_t pos = str.find(del, i);

    if(pos == string::npos){
        cout << "Error in deserialization.\n";
        return "";
    }

    string result = str.substr(i, pos - i+1);
    i = pos + 1;
    return result;
}
void show_stats_result(const string& str, int& i){
    cout << deserialize(str, i, '=');
    cout << deserialize(str, i, ' ') << endl;
    cout << deserialize(str, i, '=');
    cout << stoull(deserialize(str, i, ' ')) << endl;
    cout << deserialize(str, i, '=');
    cout << stoull(deserialize(str, i, ' ')) << endl;
    cout << deserialize(str, i, '=');
    cout << stoull(deserialize(str, i, ' ')) << endl;
    cout << deserialize(str, i, '=');
    cout << stoull(deserialize(str, i, ' ')) << endl;
    cout << deserialize(str, i, '=');
    cout << stoull(deserialize(str, i, ' ')) << endl;
    cout << deserialize(str, i, '=');
    cout << stoull(deserialize(str, i, ' ')) << endl;
    cout << deserialize(str, i, '=');
    cout << i << endl;
}

void show_result(string& response) {
    int i = 0;
    
    while (i < response.size()) {
        char type = response[i++];
        if (type == static_cast<char>(MessageType::PUT)) {
            cout << deserialize(response, i, '.') << endl;
        }
        else if (type == static_cast<char>(MessageType::GET)) { 
            show_pair_result(response, i);
        }
        else if (type == static_cast<char>(MessageType::AGG)) {
            show_pair_result(response, i);
        }
        else if (type == static_cast<char>(MessageType::FLUSH)) {
            cout << deserialize(response, i, '.') << endl;

        }
        else if (type == static_cast<char>(MessageType::QUIT)) {
            cout << deserialize(response, i, '.') << endl;
        }
        else if (type == static_cast<char>(MessageType::STATS)) {
            show_stats_result(response, i);
        }
        else if(type == static_cast<char>(MessageType::error)){
            cout << deserialize(response, i, '.') << endl;
        }
    }
}
int main() {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        cerr << "socket failed\n";
        return 1;
    }

    sockaddr_in server{};
    server.sin_family = AF_INET;
    server.sin_port = htons(kPort);

    if (inet_pton(AF_INET, kServerIp, &server.sin_addr) <= 0) {
        cerr << "inet_pton failed\n";
        return 1;
    }

    if (connect(fd, (sockaddr*)&server, sizeof(server)) < 0) {
        cerr << "connect failed\n";
        return 1;
    }

    string msg = "PUT cpu 100 10 && PUT cpu 101 11 && GET cpu 100 102";

    if (!send_with_size(fd, msg.data(), msg.size())) {
        cerr << "send failed\n";
        return 1;
    }

    string response;

    if (!recv_with_size(fd, response)) {
        cerr << "recv failed\n";
        return 1;
    }
    cout << "Server: \n";
    show_result(response);
    close(fd);
    return 0;
}
/* 
PUT cpu 1 10 && PUT temp 1 36.5 && PUT cpu 2 20 && GET cpu 0 10 && PUT cpu 1 5 && STATS cpu && GET temp 0 10 && INVALID && PUT temp 2 36.6 && GET temp 0 10 && QUIT
*/