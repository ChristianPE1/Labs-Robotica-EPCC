CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
TARGET = train

all: $(TARGET)

$(TARGET): main.cpp cartpole.hpp neural_network.hpp replay_buffer.hpp dqn_agent.hpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp

debug: CXXFLAGS = -std=c++17 -g -Wall -Wextra -DDEBUG
debug: $(TARGET)

clean:
	rm -f $(TARGET) *.csv *.png *.pdf

run: $(TARGET)
	./$(TARGET)

visualize: 
	python3 visualize.py

.PHONY: all clean run visualize debug
