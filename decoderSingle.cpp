// decoder.cpp
#include <bits/stdc++.h>
#include <cstring>
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>

using namespace std;

// ================= CONFIG =================
const int vocab_size = 27;
const int embedding_dim = 512;
const int hidden_dim = 1024;
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

/* ================= LOAD =================
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
*/

void random_matrix(Matrix &mat)
{
    srand(42);
    for (int i = 0; i < mat.rows; i++)
        for (int j = 0; j < mat.cols; j++)
            mat.m[i][j] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
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

Matrix selfAttention(const Matrix &x)
{
    auto start = chrono::high_resolution_clock::now();

    int T = x.rows;
    Matrix Q = multiply(x, Wq);
    Matrix K = multiply(x, Wk);
    Matrix V = multiply(x, Wv);

    Matrix scores = multiply(Q, transpose(K));
    for (int i = 0; i < T; i++)
        for (int j = i + 1; j < T; j++)
            scores.m[i][j] = -1e9f;
    softmax_inplace(scores);
    Matrix out = multiply(scores, V);
    Matrix result = layerNorm(add(out, x));

    auto end = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(end - start).count();
    ofstream log("single_device.log", ios::app);
    log << "selfAttention: " << ms << " ms\n";

    return result;
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

    random_matrix(token_embedding);
    random_matrix(position_embedding);
    random_matrix(Wq);
    random_matrix(Wk);
    random_matrix(Wv);
    random_matrix(W1);
    random_matrix(W2);
    random_matrix(Wout);

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
