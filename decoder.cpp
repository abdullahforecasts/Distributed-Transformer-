

// decoder.cpp
#include <bits/stdc++.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>
#include <sstream>
#include <unistd.h>

using namespace std;

// ================= CONFIG =================
const int vocab_size = 27;
const int embedding_dim = 32;
const int hidden_dim = 64;
const int block_size = 8;

// ================= VOCAB =================
vector<char> itos;

void build_vocab()
{
    for (int i = 0; i < 26; i++)
        itos.push_back('a' + i);
    itos.push_back('~');
}

int encode(char c)
{
    for (int i = 0; i < (int)itos.size(); i++)
        if (itos[i] == c)
            return i;
    return 0;
}
char decode(int i) { return itos[i]; }

// ================= MATRIX =================
struct Matrix
{
    int rows, cols;
    vector<vector<float>> m;
    Matrix(int r, int c) : rows(r), cols(c),
                           m(r, vector<float>(c, 0.0f)) {}
};

// ================= WEIGHTS =================
Matrix token_embedding(vocab_size, embedding_dim);
Matrix position_embedding(block_size, embedding_dim);
Matrix Wq(embedding_dim, embedding_dim);
Matrix Wk(embedding_dim, embedding_dim);
Matrix Wv(embedding_dim, embedding_dim);
Matrix W1(embedding_dim, hidden_dim);
Matrix W2(hidden_dim, embedding_dim);
Matrix Wout(embedding_dim, vocab_size);

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

// ================= LOAD =================
void load_matrix(Matrix &mat, const string &filename)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Cannot open: " << filename << endl;
        exit(1);
    }
    for (int i = 0; i < mat.rows; i++)
        for (int j = 0; j < mat.cols; j++)
            file >> mat.m[i][j];
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

Matrix relu(Matrix x)
{
    for (int i = 0; i < x.rows; i++)
        for (int j = 0; j < x.cols; j++)
            if (x.m[i][j] < 0)
                x.m[i][j] = 0;
    return x;
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

// ================= SPLIT / MERGE HELPERS =================
// Returns the left half (cols 0 .. half-1) of a matrix
Matrix split_left(const Matrix &src)
{
    int half = src.cols / 2;
    Matrix L(src.rows, half);
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < half; j++)
            L.m[i][j] = src.m[i][j];
    return L;
}

// Returns the right half (cols half .. cols-1) of a matrix
Matrix split_right(const Matrix &src)
{
    int half = src.cols / 2;
    int right_cols = src.cols - half;
    Matrix R(src.rows, right_cols);
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < right_cols; j++)
            R.m[i][j] = src.m[i][j + half];
    return R;
}

// Merges left (cols 0..half-1) and right (cols half..end) back into one matrix
Matrix merge_halves(const Matrix &L, const Matrix &R)
{
    int total_cols = L.cols + R.cols;
    Matrix merged(L.rows, total_cols);
    for (int i = 0; i < L.rows; i++)
    {
        for (int j = 0; j < L.cols; j++)
            merged.m[i][j] = L.m[i][j];
        for (int j = 0; j < R.cols; j++)
            merged.m[i][j + L.cols] = R.m[i][j];
    }
    return merged;
}

int connect_to_attention()
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
    {
        cerr << "socket() failed\n";
        exit(1);
    }
    sockaddr_in server{};
    server.sin_family = AF_INET;
    server.sin_port = htons(8080);
    if (inet_pton(AF_INET, "127.0.0.1", &server.sin_addr) <= 0)
    {
        cerr << "inet_pton failed\n";
        exit(1);
    }
    if (connect(fd, (sockaddr *)&server, sizeof(server)) < 0)
    {
        cerr << "connect() failed: " << strerror(errno) << "\n";
        exit(1);
    }
    return fd;
}

void send_right_half(int fd, const Matrix &Q2, const Matrix &K2,
                     const Matrix &V2, const Matrix &X2)
{
    auto send_mat = [&](const Matrix &mat)
    {
        string buf = serialize(mat.m);
        if (!send_with_size(fd, buf.data(), buf.size()))
        {
            cerr << "send_with_size failed\n";
            exit(1);
        }
    };
    send_mat(Q2);
    send_mat(K2);
    send_mat(V2);
    send_mat(X2);
}

Matrix receive_right_half(int fd)
{
    string response;
    if (!recv_with_size(fd, response))
    {
        cerr << "recv_with_size failed (attention output)\n";
        exit(1);
    }
    auto out_mat = deserialize(response);
    Matrix out(out_mat.size(), out_mat[0].size());
    out.m = out_mat;
    return out;
}

// ================= SELF-ATTENTION WITH SPLIT/OFFLOAD/MERGE =================
Matrix selfAttention(const Matrix &x)
{
    int T = x.rows;
    int half = embedding_dim / 2;

    // Full Q, K, V projections
    Matrix Q_full = multiply(x, Wq); // T x embedding_dim
    Matrix K_full = multiply(x, Wk);
    Matrix V_full = multiply(x, Wv);

    // --- Split into left (L) and right (R) halves along the col axis ---
    Matrix Q1 = split_left(Q_full);  // T x half
    Matrix Q2 = split_right(Q_full); // T x half

    Matrix K1 = split_left(K_full);
    Matrix K2 = split_right(K_full);

    Matrix V1 = split_left(V_full);
    Matrix V2 = split_right(V_full);

    // send right half to attention.cpp first
    int fd = connect_to_attention();
    send_right_half(fd, Q2, K2, V2, x);

    // --- Left half: compute attention locally ---
    Matrix scores1 = multiply(Q1, transpose(K1)); // T x T
    for (int i = 0; i < T; i++)
        for (int j = i + 1; j < T; j++)
            scores1.m[i][j] = -1e9f;
    softmax_inplace(scores1);
    Matrix out1 = multiply(scores1, V1);        // T x half
    Matrix attn_left = layerNorm(add(out1, x)); // T x half

    // receive right half result
    Matrix attn_right = receive_right_half(fd);
    close(fd);

    // --- Merge left and right attention outputs ---
    Matrix attn_merged = merge_halves(attn_left, attn_right); // T x embedding_dim

    return attn_merged;
}

Matrix FFN(const Matrix &x)
{
    Matrix y = multiply(x, W1);
    y = relu(y);
    y = multiply(y, W2);
    return layerNorm(add(y, x));
}

Matrix forward(const vector<int> &tokens)
{
    Matrix x = [&]()
    {
        int T = tokens.size();
        Matrix emb(T, embedding_dim);
        for (int i = 0; i < T; i++)
            for (int j = 0; j < embedding_dim; j++)
                emb.m[i][j] = token_embedding.m[tokens[i]][j] + position_embedding.m[i][j];
        return emb;
    }();

    x = selfAttention(x);
    x = FFN(x);

    Matrix last(1, embedding_dim);
    for (int j = 0; j < embedding_dim; j++)
        last.m[0][j] = x.m[x.rows - 1][j];

    Matrix logits = multiply(last, Wout);
    return logits;
}

// ================= GENERATE =================
string generate(int start_token, int steps)
{
    vector<int> tokens = {start_token};
    string result;

    for (int s = 0; s < steps; s++)
    {
        vector<int> ctx(tokens);
        if ((int)ctx.size() > block_size)
            ctx = vector<int>(ctx.end() - block_size, ctx.end());

        Matrix logits = forward(ctx);
        softmax_inplace(logits);

        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (logits.m[0][i] > logits.m[0][best])
                best = i;

        tokens.push_back(best);
        result += decode(best);
    }

    return result;
}

// ================= MAIN =================
int main()
{
    build_vocab();

    load_matrix(token_embedding, "token_embedding.weight.txt");
    load_matrix(position_embedding, "position_embedding.weight.txt");
    load_matrix(Wq, "Wq.weight.txt");
    load_matrix(Wk, "Wk.weight.txt");
    load_matrix(Wv, "Wv.weight.txt");
    load_matrix(W1, "W1.weight.txt");
    load_matrix(W2, "W2.weight.txt");
    load_matrix(Wout, "Wout.weight.txt");

    vector<char> test_starts = {'~', 'a', 'm', 'z'};

    // --- Single pass verification ---
    cout << "============================================================\n";
    cout << "SINGLE PASS VERIFICATION\n";
    cout << "============================================================\n";

    for (char start : test_starts)
    {
        vector<int> tokens = {encode(start)};
        Matrix logits = forward(tokens);
        softmax_inplace(logits);

        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (logits.m[0][i] > logits.m[0][best])
                best = i;

        cout << "Start='" << start << "' | predicted='" << decode(best) << "' | probs: ";
        for (int i = 0; i < vocab_size; i++)
            cout << logits.m[0][i] << " ";
        cout << "\n";
    }

    cout << "\n";

    // --- 1000-char generation ---
    cout << "============================================================\n";
    cout << "1000-CHAR GENERATION\n";
    cout << "============================================================\n";

    for (char start : test_starts)
    {
        string stream = generate(encode(start), 1000);
        cout << "\nStart='" << start << "':\n";
        for (int i = 0; i < 1000; i += 100)
            cout << stream.substr(i, 100) << "\n";
    }

    return 0;
}
