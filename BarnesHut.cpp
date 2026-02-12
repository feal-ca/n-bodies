#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <unistd.h>

using namespace std;

const double G = 1.0;
const double SOFTENING = 10.0;
const double THETA = 0.5;

struct Particle
{
    double x, y;
    double vx, vy;
    double ax, ay;
    double mass;
};

struct QuadNode {
    double x_center, y_center, side;
    double total_mass = 0, x_cm = 0, y_cm = 0;

    // -1 acts as our "NULL" or "Empty" marker
    int children[4] = {-1, -1, -1, -1}; // nw, ne, sw, se
    int particle_idx = -1; 
    bool is_leaf = true;

    // A simple constructor
    QuadNode(double x, double y, double s) 
        : x_center(x), y_center(y), side(s) {}
};


vector<double> centerQuadTree(vector<Particle> &particles)
{
    int n = particles.size();
    double xmin = particles[0].x, xmax = particles[0].x;
    double ymin = particles[0].y, ymax = particles[0].y;

    for (int i = 1; i < n; i++)
    {
        if (particles[i].x < xmin)
            xmin = particles[i].x;
        if (particles[i].x > xmax)
            xmax = particles[i].x;
        if (particles[i].y < ymin)
            ymin = particles[i].y;
        if (particles[i].y > ymax)
            ymax = particles[i].y;
    }

    double dx = xmax - xmin;
    double dy = ymax - ymin;

    double xq = xmin + 0.5 * dx;
    double yq = ymin + 0.5 * dy;
    double r = max(dx, dy) + 1.0;

    return {xq, yq, r};
}


void insert_into_quadTree(vector<QuadNode> &quadTree, int idx, Particle &p, int i)
{
    double x = p.x, y = p.y, m = p.mass;
    if (quadTree[idx].total_mass == 0 && quadTree[idx].is_leaf)
    {
        quadTree[idx].x_cm = x;
        quadTree[idx].y_cm = y;
        quadTree[idx].total_mass = m;
        quadTree[idx].particle_idx = i;
        return;
    }
    else if (!quadTree[idx].is_leaf)
    {
        double total_m = quadTree[idx].total_mass + m;
        quadTree[idx].x_cm = (quadTree[idx].x_cm * quadTree[idx].total_mass + x * m)/total_m;
        quadTree[idx].y_cm = (quadTree[idx].y_cm * quadTree[idx].total_mass + y * m)/total_m;
        quadTree[idx].total_mass = total_m;

        int q = 0;
        if (x < quadTree[idx].x_center) {
            q = (y > quadTree[idx].y_center) ? 0 : 2; // NW or SW
        } else {
            q = (y > quadTree[idx].y_center) ? 1 : 3; // NE or SE
        }
        insert_into_quadTree(quadTree, quadTree[idx].children[q], p, i);
    }

    else
    {
        // Save the particle currently occupying this leaf
        int old_idx = quadTree[idx].particle_idx;
        Particle existing_p = {quadTree[idx].x_cm, quadTree[idx].y_cm, 0, 0, 0, 0, quadTree[idx].total_mass};
        
        // Create 4 children nodes and add them to the vector
        double new_side = quadTree[idx].side / 2.0;
        double offsets_x[4] = {-new_side/2,  new_side/2, -new_side/2,  new_side/2};
        double offsets_y[4] = { new_side/2,  new_side/2, -new_side/2, -new_side/2};

        for (int i = 0; i < 4; i++) {
            quadTree[idx].children[i] = quadTree.size(); // Current size is the next index
            quadTree.push_back(QuadNode(
                quadTree[idx].x_center + offsets_x[i],
                quadTree[idx].y_center + offsets_y[i],
                new_side
            ));
        }

        quadTree[idx].is_leaf = false;
        
        // Re-insert the OLD particle and the NEW particle into the children
        // We reset total_mass temporarily so the internal node logic updates correctly
        quadTree[idx].total_mass = 0;
        quadTree[idx].particle_idx = -1;
        insert_into_quadTree(quadTree, idx, existing_p, old_idx);
        insert_into_quadTree(quadTree, idx, p, i);
    }
}

vector<QuadNode> buildQuadTree(vector<Particle> &particles)
{
    vector<QuadNode> quadTree;
    vector<double> vec = centerQuadTree(particles);
    quadTree.push_back(QuadNode(vec[0], vec[1], vec[2]));

    int n = size(particles);
    for (int i = 0; i < n; i++)
    {
        insert_into_quadTree(quadTree, 0, particles[i], i);
    }
    return quadTree;
}


double dist(double xi, double yi, double xj, double yj){
    double dx = xi - xj;
    double dy = yi - yj;
    return sqrt(dx*dx + dy*dy);
}

void updateForcesRec(double xi, double yi, double mi, vector<Particle> &particles, int i, int idx, vector<QuadNode> &quadTree)
{
    if (quadTree[idx].total_mass == 0) return;

    double r = dist(xi, yi, quadTree[idx].x_cm, quadTree[idx].y_cm);
    
    double dx = quadTree[idx].x_cm - xi;
    double dy = quadTree[idx].y_cm - yi;

    if (quadTree[idx].is_leaf && quadTree[idx].particle_idx != i)
    {
        // Use distSqr for the gravitational formula
        double distSqr = r * r + SOFTENING;
        double invDist = 1.0 / sqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;

        double f = G * mi * quadTree[idx].total_mass * invDist3;

        // FIX: Accumulate directly into the particle's acceleration
        particles[i].ax += (f * dx) / mi;
        particles[i].ay += (f * dy) / mi;
        }

    else 
    {
        double ratio = quadTree[idx].side / r;
        if (ratio < THETA) 
        {
            double distSqr = r * r + SOFTENING;
            double invDist = 1.0 / sqrt(distSqr);
            double invDist3 = invDist * invDist * invDist;

            double f = G * mi * quadTree[idx].total_mass * invDist3;

            // FIX: Accumulate directly into the particle's acceleration
            particles[i].ax += (f * dx) / mi;
            particles[i].ay += (f * dy) / mi;
        }
        else
        {
            // Recursively visit all 4 children
            for (int k = 0; k < 4; k++)
            {
                int child_idx = quadTree[idx].children[k];
                if (child_idx != -1) 
                {
                    updateForcesRec(xi, yi, mi, particles, i, child_idx, quadTree);
                }
            }
        }
    }

}

void updateForces(vector<Particle> &particles, vector<QuadNode> &quadTree)
{
    int n = particles.size();
    for (int i = 0; i < n; i++)
    {
        // IMPORTANT: Reset accelerations to zero before accumulating new forces
        particles[i].ax = 0.0;
        particles[i].ay = 0.0;
        
        double xi = particles[i].x;
        double yi = particles[i].y;
        double mi = particles[i].mass;
        
        // Only start recursion if the tree actually has nodes
        if (!quadTree.empty()) {
            updateForcesRec(xi, yi, mi, particles, i, 0, quadTree);
        }
    }
}


void updatePositions(vector<Particle> &particles, double dt)
{
    int n = particles.size();

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
    {
        particles[i].vx += particles[i].ax * dt;
        particles[i].vy += particles[i].ay * dt;
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
    }
}

void saveFrame(ofstream &file, const vector<Particle> &particles)
{
    for (size_t i = 0; i < particles.size(); i++)
    {
        file << particles[i].x << "," << particles[i].y;
        if (i < particles.size() - 1)
            file << ",";
    }
    file << "\n";
}

vector<Particle> loadParticles(const string &filename)
{
    vector<Particle> particles;
    ifstream file(filename);

    if (!file.is_open())
    {
        cerr << "Error: Could not open " << filename << endl;
        exit(1);
    }

    string line;
    while (getline(file, line))
    {
        // --- KEY CHANGE: Skip empty lines and Comments (#) ---
        if (line.empty() || line[0] == '#')
            continue;

        stringstream ss(line);
        string cell;
        Particle p;

        vector<double> values;
        while (getline(ss, cell, ','))
        {
            try
            {
                values.push_back(stod(cell));
            }
            catch (...)
            {
            }
        }

        if (values.size() >= 5)
        {
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

int main(int argc, char *argv[])
{
    // Defaults
    int steps = 1000;
    double dt = 0.05;
    int num_threads = 4;
    string input = "init.csv";
    string output = "simulation_data.csv";

    int opt;
    while ((opt = getopt(argc, argv, "t:i:o:s:d:")) != -1)
    {
        switch (opt)
        {
        case 't':
            num_threads = std::atoi(optarg);
            break;
        case 'i':
            input = optarg;
            break;
        case 'o':
            output = optarg;
            break;
        case 's':
            steps = std::atoi(optarg);
            break;
        case 'd':
            dt = std::atof(optarg);
            break; // Added dt flag
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

    if (particles.empty())
    {
        cerr << "Error: No particles loaded. Check input file." << endl;
        return 1;
    }

    cout << "Loaded " << particles.size() << " bodies." << endl;

    ofstream outFile(output);
    if (!outFile.is_open())
    {
        cerr << "Error opening output file!" << endl;
        return 1;
    }

    double startTime = omp_get_wtime();

    // Save initial state as frame 0
    saveFrame(outFile, particles);

    for (int step = 0; step < steps; step++)
    {
        vector<QuadNode> quadTree = buildQuadTree(particles);
        if (step == 0) {
        cout << "Step 0 - Particle 0 Accel: " << particles[0].ax << ", " << particles[0].ay << endl;
        }
        updateForces(particles, quadTree);
        updatePositions(particles, dt);
        saveFrame(outFile, particles);

        if (step % 100 == 0)
        {
            outFile.flush();
        }
    }

    double endTime = omp_get_wtime();
    outFile.close();

    cout << "Simulation Complete." << endl;
    cout << "Compute Time: " << (endTime - startTime) << "s" << endl;

    return 0;
}
