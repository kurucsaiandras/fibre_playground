import torch
import matplotlib.pyplot as plt
import math

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # TODO CPU is faster (?)
print(f"Using device: {device}")

# ------------------------
# Ladder setup
# ------------------------
n_points = 50
stick_length = 1.0
vertical_spacing = 1.0

# Initial centers (with a perturbation in the middle)
centers = [[0.5, r * vertical_spacing] for r in range(n_points)]
centers[n_points // 2] = [6.0, n_points // 2]  # perturb middle
centers = torch.tensor(centers, dtype=torch.float32, device=device)

# Anchors
anchor_idxs = [0, 1, n_points // 2, n_points - 2, n_points - 1]
free_idxs = [i for i in range(n_points) if i not in anchor_idxs]

# Save initial anchors (fixed)
anchor_centers = centers[anchor_idxs].detach().clone()

# Spring parameters
spring_k_linear = 5.0
spring_k_torsional = 5.0
spring_k_linear = torch.full((n_points - 1,), spring_k_linear, device=device)
spring_k_torsional = torch.full((n_points - 2,), spring_k_torsional, device=device)
spring_L_linear = torch.full((n_points - 1,), vertical_spacing, device=device)
spring_L_torsional = torch.full((n_points - 2,), math.pi, device=device)  # rest angle = pi (straight)

# Optimize only free centers
free_centers = torch.nn.Parameter(centers[free_idxs].clone().to(device))
optimizer = torch.optim.Adam([free_centers], lr=1e-2)  # much smaller lr

# ------------------------
# Helpers
# ------------------------
def build_full_params(free_centers):
    all_c = []
    it = iter(free_centers)
    for r in range(n_points):
        if r in anchor_idxs:
            idx = anchor_idxs.index(r)
            all_c.append(anchor_centers[idx])
        else:
            all_c.append(next(it))
    return torch.stack(all_c)

# Stable 2D angle computation using atan2(cross, dot)
def angle_between(p1, p2, p3, eps=1e-8):
    # vectors v1 = p1 - p2, v2 = p3 - p2
    v1 = p1 - p2
    v2 = p3 - p2
    # 2D cross product scalar
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]  # scalar
    dot = (v1 * v2).sum(dim=-1)
    # angle in [0, pi]
    angle = torch.atan2(torch.abs(cross) + eps, dot)
    # atan2(y, x) returns in (-pi, pi); because we use abs(cross) angle is in (0, pi]
    return angle

# ------------------------
# Plotting
# ------------------------
x_min, x_max = -1.0, 2.0
y_min, y_max = -1.0, n_points * vertical_spacing + 1.0

def plot_system(centers_full, step):
    coords = centers_full.detach().cpu().numpy()
    plt.clf()
    # draw linear springs between consecutive centers
    for i in range(n_points - 1):
        plt.plot([coords[i, 0], coords[i+1, 0]], [coords[i, 1], coords[i+1, 1]], "b--", lw=0.8)
    # points
    plt.scatter(coords[free_idxs, 0], coords[free_idxs, 1], c="blue", s=8)
    plt.scatter(coords[anchor_idxs, 0], coords[anchor_idxs, 1], c="red", marker="s", s=40)
    plt.title(f"Step {step}")
    plt.axis("equal")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.pause(0.001)

# ------------------------
# Optimization loop
# ------------------------
plt.ion()
fig = plt.figure()

max_grad_norm = 1.0

for step in range(5001):
    optimizer.zero_grad()
    centers_full = build_full_params(free_centers)

    # Linear spring energy (vectorized)
    diffs = centers_full[:-1] - centers_full[1:]               # (n-1, 2)
    dists = torch.norm(diffs, dim=1)
    energy_linear = 0.5 * spring_k_linear * (dists - spring_L_linear) ** 2
    energy_linear = energy_linear.sum()

    # Torsional energy (angles)
    # compute angles for triples (i, i+1, i+2) for i=0..n-3
    p1 = centers_full[:-2]
    p2 = centers_full[1:-1]
    p3 = centers_full[2:]
    angles = angle_between(p1, p2, p3)  # shape (n-2,)
    energy_torsion = 0.5 * spring_k_torsional * (angles - spring_L_torsional) ** 2
    energy_torsion = energy_torsion.sum()

    energy = energy_linear + energy_torsion

    # detect bad energy before backward
    if not torch.isfinite(energy):
        print("Non-finite energy at step", step, "-> aborting")
        break

    energy.backward()

    # gradient clipping to avoid explosion
    torch.nn.utils.clip_grad_norm_([free_centers], max_grad_norm)

    optimizer.step()

    # sanity check after step
    if not torch.all(torch.isfinite(free_centers)):
        print("Parameters became non-finite at step", step, "-> aborting")
        break

    if step % 100 == 0:
        print(f"Step {step}: Energy = {energy.item():.6f}")
        plot_system(centers_full, step)

plt.ioff()
plt.show()
