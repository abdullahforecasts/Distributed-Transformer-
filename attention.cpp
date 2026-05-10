

// attention.cpp
#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>
#include <sstream>
#include <unistd.h>
#include <bits/stdc++.h>
#include <sys/socket.h>

using namespace std;

// ================= MATRIX =================
struct Matrix
{
    int rows, cols;
    vector<vector<float>> m;
    Matrix(int r, int c) : rows(r), cols(c),
                           m(r, vector<float>(c, 0.0f)) {}
    Matrix(vector<vector<float>> &mat)
    {
        rows = mat.size();
        cols = mat[0].size();
        m = mat;
    }
};

// ================= SERIALIZE / DESERIALIZE =================
string serialize(const vector<vector<float>> &mat)
{
    string buffer;
    size_t totalSize = sizeof(int) * 2 +
                       sizeof(float) * mat.size() * mat[0].size();
    buffer.resize(totalSize);
    char *data = buffer.data();

    int rows = mat.size(), cols = mat[0].size();
    memcpy(data, &rows, sizeof(int));
    data += sizeof(int);
    memcpy(data, &cols, sizeof(int));
    data += sizeof(int);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
        {
            memcpy(data, &mat[i][j], sizeof(float));
            data += sizeof(float);
        }
    return buffer;
}

vector<vector<float>> deserialize(const string &buffer)
{
    const char *data = buffer.data();
    int rows, cols;
    memcpy(&rows, data, sizeof(int));
    data += sizeof(int);
    memcpy(&cols, data, sizeof(int));
    data += sizeof(int);

    vector<vector<float>> matrix(rows, vector<float>(cols));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
        {
            memcpy(&matrix[i][j], data, sizeof(float));
            data += sizeof(float);
        }
    return matrix;
}

// ================= SOCKET HELPERS =================
ssize_t recv_all(int socket_fd, void *data, size_t length)
{
    char *buffer = static_cast<char *>(data);
    size_t total = 0;
    while (total < length)
    {
        ssize_t r = recv(socket_fd, buffer + total, length - total, 0);
        if (r < 0)
        {
            if (errno == EINTR)
                continue;
            return -1;
        }
        if (r == 0)
            return 0;
        total += r;
    }
    return total;
}

bool send_all(int socket_fd, const void *data, size_t length)
{
    const char *buf = (const char *)data;
    size_t sent_total = 0;
    while (sent_total < length)
    {
        ssize_t s = send(socket_fd, buf + sent_total, length - sent_total, 0);
        if (s < 0)
        {
            if (errno == EINTR)
                continue;
            return false;
        }
        if (s == 0)
            return false;
        sent_total += s;
    }
    return true;
}

bool send_with_size(int socket_fd, const void *data, uint32_t length)
{
    uint32_t net = htonl(length);
    if (!send_all(socket_fd, &net, sizeof(net)))
        return false;
    return send_all(socket_fd, data, length);
}

bool recv_with_size(int socket_fd, string &out)
{
    uint32_t net = 0;
    if (recv_all(socket_fd, &net, sizeof(net)) <= 0)
        return false;
    uint32_t len = ntohl(net);
    out.resize(len);
    return recv_all(socket_fd, out.data(), len) > 0;
}

// ================= OPS =================
Matrix add(const Matrix &a, const Matrix &b)
{
    Matrix c(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < a.cols; j++)
            c.m[i][j] = a.m[i][j] + b.m[i][j];
    return c;
}

Matrix transpose(const Matrix &a)
{
    Matrix c(a.cols, a.rows);
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < a.cols; j++)
            c.m[j][i] = a.m[i][j];
    return c;
}

Matrix multiply(const Matrix &a, const Matrix &b)
{
    Matrix c(a.rows, b.cols);
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < b.cols; j++)
            for (int k = 0; k < a.cols; k++)
                c.m[i][j] += a.m[i][k] * b.m[k][j];
    return c;
}

Matrix layerNorm(Matrix x)
{
    for (int i = 0; i < x.rows; i++)
    {
        float mean = 0, var = 0;
        for (int j = 0; j < x.cols; j++)
            mean += x.m[i][j];
        mean /= x.cols;
        for (int j = 0; j < x.cols; j++)
            var += (x.m[i][j] - mean) * (x.m[i][j] - mean);
        var /= x.cols;
        for (int j = 0; j < x.cols; j++)
            x.m[i][j] = (x.m[i][j] - mean) / sqrtf(var + 1e-5f);
    }
    return x;
}

void softmax_inplace(Matrix &x)
{
    for (int i = 0; i < x.rows; i++)
    {
        float maxv = x.m[i][0];
        for (int j = 1; j < x.cols; j++)
            if (x.m[i][j] > maxv)
                maxv = x.m[i][j];
        float sum = 0;
        for (int j = 0; j < x.cols; j++)
        {
            x.m[i][j] = expf(x.m[i][j] - maxv);
            sum += x.m[i][j];
        }
        for (int j = 0; j < x.cols; j++)
            x.m[i][j] /= sum;
    }
}

// ================= RIGHT-HALF SELF-ATTENTION =================
// Receives Q2, K2, V2, X2 — computes masked self-attention + residual + layernorm
// and returns the result (same shape as X2).
Matrix selfAttention(const Matrix &x, const Matrix &Q,
                     const Matrix &K, const Matrix &V)
{
    int T = x.rows;

    Matrix scores = multiply(Q, transpose(K)); // T x T
    for (int i = 0; i < T; i++)
        for (int j = i + 1; j < T; j++)
            scores.m[i][j] = -1e9f;

    softmax_inplace(scores);
    Matrix out = multiply(scores, V); // T x half
    return layerNorm(add(out, x));    // residual + norm
}

// ================= MAIN =================
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "usage: attention <port>\n";
        return 1;
    }
    int port = stoi(argv[1]);

    int server_fd, client_fd;
    sockaddr_in server_addr{}, client_addr{};
    socklen_t client_len = sizeof(client_addr);

    // 1. Create socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0)
    {
        cerr << "socket() failed\n";
        return 1;
    }

    // Allow quick rebind after restart
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // 2. Bind
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    if (bind(server_fd, (sockaddr *)&server_addr, sizeof(server_addr)) < 0)
    {
        cerr << "bind() failed: " << strerror(errno) << "\n";
        return 1;
    }

    // 3. Listen
    if (listen(server_fd, 10) < 0)
    {
        cerr << "listen() failed\n";
        return 1;
    }

    cout << "[attention] Server listening on port 8080...\n";

    // 4. Accept connections in a loop (one per forward() call from decoder)
    while (true)
    {
        client_fd = accept(server_fd, (sockaddr *)&client_addr, &client_len);
        if (client_fd < 0)
        {
            cerr << "accept() failed: " << strerror(errno) << "\n";
            continue;
        }
        cout << "[attention] Client connected.\n";

        // ---- Receive Q2 ----
        string buf;
        if (!recv_with_size(client_fd, buf))
        {
            cerr << "[attention] Failed to receive Q2\n";
            close(client_fd);
            continue;
        }
        auto Q_mat = deserialize(buf);
        Matrix Q(Q_mat);

        // ---- Receive K2 ----
        buf.clear();
        if (!recv_with_size(client_fd, buf))
        {
            cerr << "[attention] Failed to receive K2\n";
            close(client_fd);
            continue;
        }
        auto K_mat = deserialize(buf);
        Matrix K(K_mat);

        // ---- Receive V2 ----
        buf.clear();
        if (!recv_with_size(client_fd, buf))
        {
            cerr << "[attention] Failed to receive V2\n";
            close(client_fd);
            continue;
        }
        auto V_mat = deserialize(buf);
        Matrix V(V_mat);

        // ---- Receive x ----
        buf.clear();
        if (!recv_with_size(client_fd, buf))
        {
            cerr << "[attention] Failed to receive X2\n";
            close(client_fd);
            continue;
        }
        auto X_mat = deserialize(buf);
        Matrix X(X_mat);

        cout << "[attention] Received Q2 (" << Q.rows << "x" << Q.cols << "), "
             << "K2 (" << K.rows << "x" << K.cols << "), "
             << "V2 (" << V.rows << "x" << V.cols << "), "
             << "X2 (" << X.rows << "x" << X.cols << ").\n";

        /* ---- Wait 2 seconds, doing nothing ----
        cout << "[attention] Holding for 2 seconds...\n";
        sleep(2);
        cout << "[attention] Done waiting. Computing right-half attention...\n";
        */

        // ---- Compute right-half self-attention ----
        Matrix result = selfAttention(X, Q, K, V);

        // ---- Send result back to decoder ----
        string out_buf = serialize(result.m);
        if (!send_with_size(client_fd, out_buf.data(), out_buf.size()))
        {
            cerr << "[attention] Failed to send result\n";
        }
        else
        {
            cout << "[attention] Sent attention output ("
                 << result.rows << "x" << result.cols << ") back to decoder.\n";
        }

        close(client_fd);
    }

    close(server_fd);
    return 0;
}