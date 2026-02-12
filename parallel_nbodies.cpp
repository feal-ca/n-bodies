#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <unistd.h>

using namespace std;

const double G = 1.0;
const double SOFTENING = 2.0;

struct Particle {
    double x, y;
    double vx, vy;
    double ax, ay;
    double mass;
};

void updateForces(vector<Particle>& particles) {
    int n = particles.size();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double fx = 0.0;
        double fy = 0.0;

        // Pre-calculate i-th particle constants to reduce memory lookups
        double xi = particles[i].x;
        double yi = particles[i].y;
        double mi = particles[i].mass;

        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            double dx = particles[j].x - xi;
            double dy = particles[j].y - yi;
            double distSqr = dx*dx + dy*dy + SOFTENING;

            double invDist = 1.0 / sqrt(distSqr);
            double invDist3 = invDist * invDist * invDist;

            double f = G * mi * particles[j].mass * invDist3;

            fx += f * dx;
            fy += f * dy;
        }

        particles[i].ax = fx / mi;
        particles[i].ay = fy / mi;
    }
}

void updatePositions(vector<Particle>& particles, double dt) {
    int n = particles.size();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        particles[i].vx += particles[i].ax * dt;
        particles[i].vy += particles[i].ay * dt;
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
    }
}

void saveFrame(ofstream& file, const vector<Particle>& particles) {
    for (size_t i = 0; i < particles.size(); i++) {
        file << particles[i].x << "," << particles[i].y;
        if (i < particles.size() - 1) file << ",";
    }
    file << "\n";
}

vector<Particle> loadParticles(const string& filename) {
    vector<Particle> particles;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Could not open " << filename << endl;
        exit(1);
    }

    string line;
    while (getline(file, line)) {
        // --- KEY CHANGE: Skip empty lines and Comments (#) ---
        if (line.empty() || line[0] == '#') continue;

        stringstream ss(line);
        string cell;
        Particle p;

        vector<double> values;
        while (getline(ss, cell, ',')) {
            try {
                values.push_back(stod(cell));
            } catch (...) {
            }
        }

        if (values.size() >= 5) {
            p.x = values[0];
            p.y = values[1];
            p.vx = values[2];
            p.vy = values[3];
            p.mass = values[4];
            p.ax = 0;
            p.ay = 0;
            particles.push_back(p);
        }
    }
    return particles;
}

int main(int argc, char* argv[]) {
    // Defaults
    int steps = 1000;
    double dt = 0.05;
    int num_threads = 4;
    string input = "init.csv";
    string output = "simulation_data.csv";

    int opt;
    while ((opt = getopt(argc, argv, "t:i:o:s:d:")) != -1) {
        switch (opt) {
            case 't': num_threads = std::atoi(optarg); break;
            case 'i': input = optarg; break;
            case 'o': output = optarg; break;
            case 's': steps = std::atoi(optarg); break;
            case 'd': dt = std::atof(optarg); break; // Added dt flag
            default:
                std::cerr << "Usage: " << argv[0] << " [-t threads] [-i input] [-o output] [-s steps] [-d dt]\n";
                return 1;
        }
    }

    omp_set_num_threads(num_threads);
    cout << "--- N-Body C++ Simulation ---" << endl;
    cout << "Threads: " << num_threads << endl;
    cout << "Input:   " << input << endl;
    cout << "Output:  " << output << endl;

    vector<Particle> particles = loadParticles(input);

    if (particles.empty()) {
        cerr << "Error: No particles loaded. Check input file." << endl;
        return 1;
    }

    cout << "Loaded " << particles.size() << " bodies." << endl;

    ofstream outFile(output);
    if (!outFile.is_open()) {
        cerr << "Error opening output file!" << endl;
        return 1;
    }

    double startTime = omp_get_wtime();

    // Save initial state as frame 0
    saveFrame(outFile, particles);

    for (int step = 0; step < steps; step++) {
        updateForces(particles);
        updatePositions(particles, dt);
        saveFrame(outFile, particles);

        if (step % 100 == 0) {
            outFile.flush();
        }
    }

    double endTime = omp_get_wtime();
    outFile.close();

    cout << "Simulation Complete." << endl;
    cout << "Compute Time: " << (endTime - startTime) << "s" << endl;

    return 0;
}
