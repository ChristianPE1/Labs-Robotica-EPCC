#ifndef REPLAY_BUFFER_HPP
#define REPLAY_BUFFER_HPP

#include <vector>
#include <deque>
#include <random>
#include <array>

// Transición del entorno
struct Transition {
    std::vector<double> state;
    int action;
    double reward;
    std::vector<double> next_state;
    bool done;
    
    Transition(const std::vector<double>& s, int a, double r, 
               const std::vector<double>& ns, bool d)
        : state(s), action(a), reward(r), next_state(ns), done(d) {}
};

// Buffer de replay para experience replay
class ReplayBuffer {
private:
    std::deque<Transition> buffer;
    size_t capacity;
    std::mt19937 rng;

public:
    ReplayBuffer(size_t cap, unsigned int seed = 42) 
        : capacity(cap), rng(seed) {}
    
    void push(const std::vector<double>& state, int action, double reward,
              const std::vector<double>& next_state, bool done) {
        if (buffer.size() >= capacity) {
            buffer.pop_front();
        }
        buffer.emplace_back(state, action, reward, next_state, done);
    }
    
    std::vector<Transition> sample(size_t batch_size) {
        std::vector<Transition> batch;
        std::uniform_int_distribution<size_t> dist(0, buffer.size() - 1);
        
        for (size_t i = 0; i < batch_size; ++i) {
            batch.push_back(buffer[dist(rng)]);
        }
        
        return batch;
    }
    
    size_t size() const { return buffer.size(); }
    bool canSample(size_t batch_size) const { return buffer.size() >= batch_size; }
};

#endif // REPLAY_BUFFER_HPP
