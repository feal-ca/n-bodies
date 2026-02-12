import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.patches as patches
import numpy as np
import csv
import argparse
import math
import json

# ================= Argument Parsing =================
parser = argparse.ArgumentParser(description="N-Body Galaxy Architect")
parser.add_argument("-i", "--input", help="Load existing CSV file")
parser.add_argument("-o", "--output", default="init.csv", help="Output filename")
args = parser.parse_args()

# ================= Configuration =================
SPACE_LIMIT = 1000
VEL_SCALE = 10.0
BG_COLOR = '#050505'
SLIDER_BG = '#1a1a1a'
TEXT_COLOR = 'white'
HIGHLIGHT_COLOR = '#FFD700'
HANDLE_COLOR = '#FF3333'

# ================= Galaxy Class =================
class GalaxyClump:
    def __init__(self, ax, x, y, vx, vy, count, radius,
                 mass_avg, mass_std, speed_std, orbit_power):
        self.ax = ax
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.count = int(count)
        self.radius = radius
        self.mass_avg = mass_avg
        self.mass_std = mass_std
        self.speed_std = speed_std
        self.orbit_power = orbit_power

        self.scatter = None
        self.ring = None
        self.arrow = None
        self.v_handle = None

        self.particles = []
        self.regenerate()

    def get_params(self, uid):
        return {
            "uid": uid,
            "x": self.x, "y": self.y,
            "vx": self.vx, "vy": self.vy,
            "count": self.count,
            "radius": self.radius,
            "mass_avg": self.mass_avg,
            "mass_std": self.mass_std,
            "speed_std": self.speed_std,
            "orbit_power": self.orbit_power
        }

    def regenerate(self):
        if self.count <= 0:
            self.particles = []
            return

        px = np.random.normal(self.x, self.radius, self.count)
        py = np.random.normal(self.y, self.radius, self.count)

        rx, ry = px - self.x, py - self.y
        dist = np.maximum(np.sqrt(rx**2 + ry**2), 1.0)

        tan_x, tan_y = -ry / dist, rx / dist
        v_mag = self.orbit_power * (15.0 / np.sqrt(dist))

        pvx = self.vx + tan_x * v_mag + np.random.normal(0, self.speed_std, self.count)
        pvy = self.vy + tan_y * v_mag + np.random.normal(0, self.speed_std, self.count)

        pmass = np.maximum(np.random.normal(self.mass_avg, self.mass_std, self.count), 0.1)

        self.particles = np.column_stack((px, py, pvx, pvy, pmass))

    def draw(self):
        if self.scatter:
            self.scatter.remove()

        if len(self.particles):
            sizes = np.clip(2 + self.particles[:, 4] / 2.0, 1, 50)
            self.scatter = self.ax.scatter(
                self.particles[:, 0],
                self.particles[:, 1],
                s=sizes,
                c='white',
                alpha=0.7
            )

        if self.ring:
            self.update_selection_graphics()

    def clear_selection_graphics(self):
        for obj in (self.ring, self.arrow, self.v_handle):
            if obj:
                obj.remove()
        self.ring = self.arrow = self.v_handle = None

    def update_selection_graphics(self):
        self.clear_selection_graphics()

        self.ring = patches.Circle(
            (self.x, self.y), self.radius,
            edgecolor=HIGHLIGHT_COLOR,
            facecolor='none',
            linestyle='--',
            linewidth=1.5
        )
        self.ax.add_patch(self.ring)

        self.arrow = self.ax.arrow(
            self.x, self.y,
            self.vx * VEL_SCALE,
            self.vy * VEL_SCALE,
            head_width=20,
            head_length=30,
            fc=HIGHLIGHT_COLOR,
            ec=HIGHLIGHT_COLOR
        )

        tip_x = self.x + self.vx * VEL_SCALE
        tip_y = self.y + self.vy * VEL_SCALE
        self.v_handle = self.ax.scatter(
            [tip_x], [tip_y],
            s=100,
            c=HANDLE_COLOR,
            edgecolors='white',
            zorder=10
        )

    def set_selected(self, state):
        if state:
            self.update_selection_graphics()
        else:
            self.clear_selection_graphics()

    def update_pos(self, x, y):
        dx, dy = x - self.x, y - self.y
        self.x, self.y = x, y
        self.particles[:, 0] += dx
        self.particles[:, 1] += dy
        self.draw()

    def update_vel_from_handle(self, hx, hy):
        new_vx = (hx - self.x) / VEL_SCALE
        new_vy = (hy - self.y) / VEL_SCALE
        dvx, dvy = new_vx - self.vx, new_vy - self.vy
        self.vx, self.vy = new_vx, new_vy
        self.particles[:, 2] += dvx
        self.particles[:, 3] += dvy
        self.draw()

    def remove(self):
        if self.scatter:
            self.scatter.remove()
        self.clear_selection_graphics()

# ================= Global State =================
clumps = []
selected_idx = -1
radio = None
mode = "IDLE"
drag_start = None

# ================= UI Setup =================
fig = plt.figure(figsize=(15, 9), facecolor=BG_COLOR)
fig.canvas.manager.set_window_title(f"Galaxy Architect - {args.output}")

ax_main = plt.axes([0.05, 0.05, 0.60, 0.9])
ax_sidebar = plt.axes([0.68, 0.05, 0.28, 0.9], facecolor=BG_COLOR)
ax_sidebar.axis('off')

ax_main.set_facecolor('black')
ax_main.set_xlim(-SPACE_LIMIT, SPACE_LIMIT)
ax_main.set_ylim(-SPACE_LIMIT, SPACE_LIMIT)
ax_main.set_aspect('equal')
ax_main.grid(True, color='#333333', linestyle=':')

# ================= Sliders =================
sliders = {}

def add_slider(label, y, vmin, vmax, val, fmt="%1.1f"):
    ax = plt.axes([0.75, y, 0.18, 0.02], facecolor=SLIDER_BG)
    s = Slider(ax, label, vmin, vmax, valinit=val, valfmt=fmt)
    s.label.set_color(TEXT_COLOR)
    s.valtext.set_color(TEXT_COLOR)
    return s

sliders['count'] = add_slider('Count', 0.92, 1, 1000, 200, "%d")
sliders['radius'] = add_slider('Radius', 0.88, 10, 500, 100)
sliders['mass'] = add_slider('Mass (10^x)', 0.84, 0, 10, 1)
sliders['m_std'] = add_slider('Mass Var', 0.80, 0, 50, 1)
sliders['spin'] = add_slider('Spin', 0.76, -100, 100, 0)
sliders['chaos'] = add_slider('Chaos', 0.72, 0, 10, 0.5)

# ================= Galaxy List =================
ax_list = plt.axes([0.70, 0.40, 0.25, 0.22], facecolor=SLIDER_BG)

def rebuild_radio():
    global radio
    ax_list.clear()
    ax_list.set_facecolor(SLIDER_BG)

    if not clumps:
        radio = None
        fig.canvas.draw_idle()
        return

    labels = [f"Galaxy {i+1} (N={c.count})" for i, c in enumerate(clumps)]
    active = selected_idx if selected_idx != -1 else 0

    radio = RadioButtons(ax_list, labels, active=active)
    for lbl in radio.labels:
        lbl.set_color('white')
        lbl.set_fontsize(9)

    radio.on_clicked(on_radio_select)
    fig.canvas.draw_idle()

def on_radio_select(label):
    idx = int(label.split()[1]) - 1
    select_clump(idx)

# ================= Buttons =================
ax_del = plt.axes([0.70, 0.32, 0.10, 0.05])
btn_del = Button(ax_del, 'DELETE', color='#8B0000', hovercolor='#FF0000')
btn_del.label.set_color('white')

ax_save = plt.axes([0.82, 0.32, 0.10, 0.05])
btn_save = Button(ax_save, 'SAVE', color='#006400', hovercolor='#00FF00')
btn_save.label.set_color('white')

def delete_selected(event):
    global selected_idx
    if selected_idx == -1:
        return
    clumps[selected_idx].remove()
    del clumps[selected_idx]
    selected_idx = -1
    rebuild_radio()
    fig.canvas.draw_idle()

def save_data(event):
    if not clumps:
        return

    header_lines = []
    all_data = []

    for i, c in enumerate(clumps):
        header_lines.append(f"# CONFIG:{json.dumps(c.get_params(i))}\n")
        ids = np.full((len(c.particles), 1), i)
        all_data.append(np.hstack([c.particles, ids]))

    with open(args.output, "w", newline="") as f:
        f.writelines(header_lines)
        writer = csv.writer(f)
        writer.writerows(np.vstack(all_data))

btn_del.on_clicked(delete_selected)
btn_save.on_clicked(save_data)

# ================= Selection & Sliders =================
def select_clump(idx):
    global selected_idx
    if selected_idx != -1:
        clumps[selected_idx].set_selected(False)
    selected_idx = idx
    if idx != -1:
        clumps[idx].set_selected(True)
        set_sliders_from_clump(clumps[idx])
    fig.canvas.draw_idle()

def set_sliders_from_clump(c):
    for s in sliders.values():
        s.eventson = False
    sliders['count'].set_val(c.count)
    sliders['radius'].set_val(c.radius)
    sliders['mass'].set_val(math.log10(max(c.mass_avg, 0.1)))
    sliders['m_std'].set_val(c.mass_std)
    sliders['spin'].set_val(c.orbit_power)
    sliders['chaos'].set_val(c.speed_std)
    for s in sliders.values():
        s.eventson = True

def update_selected_from_sliders(val):
    if selected_idx == -1:
        return
    c = clumps[selected_idx]
    c.count = int(sliders['count'].val)
    c.radius = sliders['radius'].val
    c.mass_avg = 10 ** sliders['mass'].val
    c.mass_std = sliders['m_std'].val
    c.orbit_power = sliders['spin'].val
    c.speed_std = sliders['chaos'].val
    c.regenerate()
    c.draw()
    rebuild_radio()

for s in sliders.values():
    s.on_changed(update_selected_from_sliders)

# ================= Mouse Interaction =================
def on_press(event):
    global mode, drag_start
    if event.inaxes != ax_main:
        return
    drag_start = (event.xdata, event.ydata)

    if selected_idx != -1:
        c = clumps[selected_idx]
        if c.v_handle:
            hx = c.x + c.vx * VEL_SCALE
            hy = c.y + c.vy * VEL_SCALE
            if np.hypot(hx - event.xdata, hy - event.ydata) < 60:
                mode = "DRAG_VEL"
                return

    for i, c in enumerate(clumps):
        if np.hypot(c.x - event.xdata, c.y - event.ydata) < c.radius * 0.5:
            select_clump(i)
            mode = "DRAG_POS"
            return

    select_clump(-1)
    mode = "CREATE"

def on_drag(event):
    if event.inaxes != ax_main or drag_start is None:
        return
    if mode == "DRAG_POS" and selected_idx != -1:
        clumps[selected_idx].update_pos(event.xdata, event.ydata)
    elif mode == "DRAG_VEL" and selected_idx != -1:
        clumps[selected_idx].update_vel_from_handle(event.xdata, event.ydata)
    fig.canvas.draw_idle()

def on_release(event):
    global mode, drag_start
    if event.inaxes != ax_main or drag_start is None:
        return

    if mode == "CREATE":
        dx = event.xdata - drag_start[0]
        dy = event.ydata - drag_start[1]
        if np.hypot(dx, dy) > 20:
            c = GalaxyClump(
                ax_main,
                drag_start[0], drag_start[1],
                dx * 0.5, dy * 0.5,
                sliders['count'].val,
                sliders['radius'].val,
                10 ** sliders['mass'].val,
                sliders['m_std'].val,
                sliders['chaos'].val,
                sliders['spin'].val
            )
            clumps.append(c)
            c.draw()
            select_clump(len(clumps) - 1)
            rebuild_radio()

    mode = "IDLE"
    drag_start = None
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_drag)
fig.canvas.mpl_connect('button_release_event', on_release)

# ================= Loader =================
def load_from_csv(filename):
    configs = {}
    raw = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("# CONFIG:"):
                cfg = json.loads(line.replace("# CONFIG:", "").strip())
                configs[cfg["uid"]] = cfg
            elif line.strip() and not line.startswith("#"):
                raw.append(line)

    if not raw:
        return

    data = np.loadtxt(raw, delimiter=',')

    for uid in np.unique(data[:, 5]).astype(int):
        subset = data[data[:, 5] == uid][:, :5]
        cfg = configs[uid]
        c = GalaxyClump(
            ax_main,
            cfg["x"], cfg["y"],
            cfg["vx"], cfg["vy"],
            cfg["count"], cfg["radius"],
            cfg["mass_avg"], cfg["mass_std"],
            cfg["speed_std"], cfg["orbit_power"]
        )
        c.particles = subset
        c.draw()
        clumps.append(c)

    if clumps:
        select_clump(0)
        rebuild_radio()

if args.input:
    load_from_csv(args.input)

plt.show()
