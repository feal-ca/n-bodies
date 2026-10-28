# n-bodies

Two 2D N-body codes in C++ with OpenMP: direct summation, which is O(n²),
and a Barnes-Hut quadtree, which is O(n log n). A pair project at UPC with
[FernandoAlosno](https://github.com/FernandoAlosno). There's a short
write-up on the [project page](https://feal-ca.github.io/work/n-body/).

We timed both on the same 112-thread node, for N from 100 to 100,000. At
N = 10,000, direct summation ran about 33× faster on 112 threads than on
one. Barnes-Hut does less work but never got past about 7×, and had mostly
stopped improving by 32 threads. Its traversal is irregular, its memory
access is scattered, and there's less independent work to split between
threads.

## Files

- `parallel_nbodies.cpp`: direct summation.
- `BarnesHut.cpp`: the quadtree version, with opening angle θ = 0.5.
- `painter.py`: an interactive Matplotlib editor for initial conditions.
  You place clumps of bodies, set their count, radius, mass and spin with
  sliders, and save them as `init.csv`.
- `visualize_nbodies.py`: animates a run as a scatter plot over a
  mass-weighted density map. `visualize_nbodies2.py` is the same with a
  frame counter.

## Running it

```sh
g++ -O3 -fopenmp parallel_nbodies.cpp -o nbody
g++ -O3 -fopenmp BarnesHut.cpp -o barneshut

python painter.py -o init.csv
./barneshut -t 8 -i init.csv -o simulation_data.csv -s 1000 -d 0.05
python visualize_nbodies.py -i init.csv -s simulation_data.csv
```

Both programs take the same flags: `-t` threads (default 4), `-i` input
(`init.csv`), `-o` output (`simulation_data.csv`), `-s` steps (1000) and
`-d` time step (0.05). They print the compute time when they finish.

The input has one body per line as `x,y,vx,vy,mass`, and lines starting with
`#` are skipped. The output has one line per step with `x,y` for every body,
in input order.

G = 1, and each step updates the velocities and then the positions
(semi-implicit Euler). The softening added to r² is 2 in the direct code and
10 in Barnes-Hut, so the two won't give identical trajectories from the same
input.
