#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
#include <chrono>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

class Graph {
public:
    int V;
    vector<vector<int>> adj;

    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void generateRandomGraph(int edges);
    void sequentialBFS(int start);
    void parallelBFS(int start);
    void sequentialDFS(int start);
    void parallelDFS(int start);
};

void Graph::generateRandomGraph(int edges) {
    srand(time(0));
    for (int i = 0; i < edges; i++) {
        int u = rand() % V;
        int v = rand() % V;
        if (u != v) {
            addEdge(u, v);
        }
    }
}

void Graph::sequentialBFS(int start) {
    vector<bool> visited(V, false);
    queue<int> q;
    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int node = q.front();
        q.pop();

        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}

void Graph::parallelBFS(int start) {
    vector<bool> visited(V, false);
    queue<int> q;
    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int size = q.size();
        vector<int> levelNodes;
        
        #pragma omp parallel for shared(visited, q) 
        for (int i = 0; i < size; i++) {
            int node;
            #pragma omp critical
            {
                if (!q.empty()) {
                    node = q.front();
                    q.pop();
                }
            }
            for (int neighbor : adj[node]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    #pragma omp critical
                    levelNodes.push_back(neighbor);
                }
            }
        }
        
        for (int node : levelNodes) {
            q.push(node);
        }
    }
}

void Graph::sequentialDFS(int start) {
    vector<bool> visited(V, false);
    stack<int> s;
    s.push(start);

    while (!s.empty()) {
        int node = s.top();
        s.pop();
        if (!visited[node]) {
            visited[node] = true;
            for (int neighbor : adj[node]) {
                if (!visited[neighbor]) {
                    s.push(neighbor);
                }
            }
        }
    }
}

void Graph::parallelDFS(int start) {
    vector<bool> visited(V, false);
    stack<int> s;
    s.push(start);

    #pragma omp parallel
    {
        while (!s.empty()) {
            int node;
            #pragma omp critical
            {
                if (!s.empty()) {
                    node = s.top();
                    s.pop();
                }
            }
            if (!visited[node]) {
                visited[node] = true;
                #pragma omp parallel for
                for (int i = 0; i < adj[node].size(); i++) {
                    int neighbor = adj[node][i];
                    if (!visited[neighbor]) {
                        #pragma omp critical
                        s.push(neighbor);
                    }
                }
            }
        }
    }
}

int main() {
    int V, E;
    
    cout << "Enter the number of vertices: ";
    cin >> V;
    
    cout << "Enter the number of edges: ";
    cin >> E;

    if (E > (V * (V - 1)) / 2) {
        cout << "Too many edges for the given number of vertices. Adjusting to maximum possible edges.\n";
        E = (V * (V - 1)) / 2;
    }

    Graph g(V);
    g.generateRandomGraph(E);

    auto start = high_resolution_clock::now();
    g.sequentialBFS(0);
    auto stop = high_resolution_clock::now();
    auto seqBFS_time = duration_cast<microseconds>(stop - start);

    start = high_resolution_clock::now();
    g.parallelBFS(0);
    stop = high_resolution_clock::now();
    auto parBFS_time = duration_cast<microseconds>(stop - start);

    start = high_resolution_clock::now();
    g.sequentialDFS(0);
    stop = high_resolution_clock::now();
    auto seqDFS_time = duration_cast<microseconds>(stop - start);

    start = high_resolution_clock::now();
    g.parallelDFS(0);
    stop = high_resolution_clock::now();
    auto parDFS_time = duration_cast<microseconds>(stop - start);

    cout << "Sequential BFS Time: " << seqBFS_time.count() << " microseconds" << endl;
    cout << "Parallel BFS Time: " << parBFS_time.count() << " microseconds" << endl;
    cout << "Speedup for BFS: " << (double)seqBFS_time.count() / parBFS_time.count() << endl;

    cout << "Sequential DFS Time: " << seqDFS_time.count() << " microseconds" << endl;
    cout << "Parallel DFS Time: " << parDFS_time.count() << " microseconds" << endl;
    cout << "Speedup for DFS: " << (double)seqDFS_time.count() / parDFS_time.count() << endl;

    return 0;
}
