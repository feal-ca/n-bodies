import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import csv
import numpy as np
import argparse

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="N-Body Visualizer")
parser.add_argument("-i", "--init", default="init.csv", help="Initial conditions file (for masses)")
parser.add_argument("-s", "--sim", default="simulation_data.csv", help="Simulation data file")
parser.add_argument("-r", "--res", type=int, default=150, help="Grid resolution for heatmap (default: 150)")
args = parser.parse_args()

SIM_FILE = args.sim
INIT_FILE = args.init
GRID_RES = args.res
LIMIT = 1000

# --- 1. Load Masses ---
print(f"Reading masses from {INIT_FILE}...")
masses = []
try:
    with open(INIT_FILE, 'r') as f:
        # We handle the file manually to skip comments
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            row = line.strip().split(',')
            # Expecting x, y, vx, vy, mass, [id]
            if len(row) >= 5:
                try:
                    masses.append(float(row[4]))
                except ValueError:
                    pass
except FileNotFoundError:
    print(f"Error: {INIT_FILE} not found.")
    exit()

if not masses:
    print("Error: No masses loaded. Check init file format.")
    exit()

masses = np.array(masses)
print(f"Loaded {len(masses)} bodies.")

# --- 2. Load Simulation Steps ---
print(f"Reading simulation positions from {SIM_FILE}...")
frames = []
try:
    with open(SIM_FILE, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            row = line.strip().split(',')
            try:
                coords = [float(x) for x in row]
                # Reshape to (N, 2)
                # We check if the frame size matches the number of masses we loaded
                frame_data = np.array(coords).reshape(-1, 2)

                # Sanity check: does particle count match?
                if len(frame_data) == len(masses):
                    frames.append(frame_data)
                else:
                    # If mismatch, it might be a partial write or corruption
                    pass
            except ValueError:
                continue

except FileNotFoundError:
    print(f"Error: {SIM_FILE} not found.")
    exit()

if not frames:
    print("Error: No simulation data found.")
    exit()

print(f"Loaded {len(frames)} frames.")

# --- 3. Setup Plot ---
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

extent = [-LIMIT, LIMIT, -LIMIT, LIMIT]

# --- Visualization 1: Scatter Plot ---
ax1.set_xlim(-LIMIT, LIMIT)
ax1.set_ylim(-LIMIT, LIMIT)
ax1.set_aspect('equal')
ax1.set_title("Particle Scatter")

# Normalize mass colors
# Vmin/Vmax protection against 0 mass
norm_mass = mcolors.LogNorm(vmin=max(masses.min(), 1e-6), vmax=masses.max())
cmap_scatter = plt.get_cmap('plasma')

scatter = ax1.scatter(frames[0][:, 0], frames[0][:, 1],
                      s=5, c=masses, cmap=cmap_scatter, norm=norm_mass, alpha=0.8)

cbar1 = plt.colorbar(scatter, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Mass (Log Scale)')

# --- Visualization 2: Mass Density Heatmap ---
ax2.set_xlim(-LIMIT, LIMIT)
ax2.set_ylim(-LIMIT, LIMIT)
ax2.set_aspect('equal')
ax2.set_title("Mass Density Heatmap")

# Heatmap Scaling
heatmap_norm = mcolors.LogNorm(vmin=1e-3, vmax=masses.sum()/5.0)
cmap_heat = plt.get_cmap('inferno')

empty_grid = np.zeros((GRID_RES, GRID_RES))
img = ax2.imshow(empty_grid, origin='lower', extent=extent,
                 cmap=cmap_heat, norm=heatmap_norm, interpolation='nearest')

cbar2 = plt.colorbar(img, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Mass Density')

frame_text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, color='white', fontsize=12)

# --- Update Function ---
def update(frame_idx):
    current_data = frames[frame_idx]
    
    # Update Scatter
    scatter.set_offsets(current_data)
    
    # Update Heatmap
    hist, _, _ = np.histogram2d(current_data[:, 0], current_data[:, 1], bins=GRID_RES,
                                range=[[-LIMIT, LIMIT], [-LIMIT, LIMIT]],
                                weights=masses)
    img.set_data(hist.T)

    # 2. UPDATE THE TEXT CONTENT
    frame_text.set_text(f"Frame: {frame_idx}")

    # 3. RETURN THE TEXT OBJECT (Critical for blit=True!)
    return scatter, img, frame_text 

# Animation
ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=20, blit=True)

plt.tight_layout()
plt.show()
