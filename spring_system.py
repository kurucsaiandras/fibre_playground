import torch
import matplotlib.pyplot as plt

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
device = torch.device("cpu")  # TODO CPU is faster (?)

# TODO: idea: have one spring for length, one for angle (can be ported to 3D more easily)

# ------------------------
# Ladder setup
# ------------------------
n_rungs = 50
stick_length = 1.0  # horizontal bar length
vertical_spacing = 1.0

# Parameters: centers + orientations
centers = []
thetas = []
for r in range(n_rungs):
    centers.append([0.5, r * vertical_spacing])  # center between left/right
    thetas.append(0.0)                           # horizontal initially

centers[n_rungs//2] = [6.0, n_rungs//2]  # perturb middle rung

#thetas[0] = 0.5  # slight tilt at bottom
#thetas[-1] = -0.5  # slight tilt at top
#centers[0] = [0.5, 0.0+0.5]
#centers[-1] = [0.5, (n_rungs-1)*vertical_spacing-0.5]

centers = torch.tensor(centers, dtype=torch.float32, device=device, requires_grad=True)
thetas = torch.tensor(thetas, dtype=torch.float32, device=device, requires_grad=True)
# Anchors: fixed points (index into coords)
anchor_idxs = [0, n_rungs//2, n_rungs-1]  # first and last rung
free_idxs = [i for i in range(n_rungs) if i not in anchor_idxs]

# Anchor and free indeces into coords
# Each rung has two endpoints: left, right
anchor_coord_idxs = [2*i for i in anchor_idxs] + [2*i+1 for i in anchor_idxs]
free_coord_idxs = [2*i for i in free_idxs] + [2*i+1 for i in free_idxs]

# Save initial anchors
anchor_centers = centers[anchor_idxs].detach().clone()
anchor_thetas  = thetas[anchor_idxs].detach().clone()

# Springs: vertical connections between consecutive rungs
springs = []
for r in range(n_rungs-1):
    springs.append((r, "left",  r+1, "left"))
    springs.append((r, "right", r+1, "right"))
# Optionally, add diagonal springs for stability
for r in range(n_rungs-1):
    springs.append((r, "left",  r+1, "right"))
    springs.append((r, "right", r+1, "left"))

spring_L_sides = [vertical_spacing]*(2*(n_rungs-1))  # rest lengths
spring_L_diags = [(vertical_spacing**2 + stick_length**2)**0.5]*(2*(n_rungs-1))
spring_L = torch.tensor(spring_L_sides + spring_L_diags, device=device)
spring_k = torch.tensor([100.0]*len(springs), device=device)

# Remove anchors from optimization
free_centers = torch.nn.Parameter(centers[free_idxs].clone().to(device))
free_thetas  = torch.nn.Parameter(thetas[free_idxs].clone().to(device))
optimizer = torch.optim.Adam([free_centers, free_thetas], lr=0.1)


# ------------------------
# Helper: get endpoints
# ------------------------
def rung_endpoints(center, theta, L=stick_length):
    dx = 0.5 * L * torch.cos(theta)
    dy = 0.5 * L * torch.sin(theta)
    left  = center - torch.stack([dx, dy])
    right = center + torch.stack([dx, dy])
    return left, right

def build_full_params(free_centers, free_thetas):
    all_c = []
    all_t = []
    free_iter_c = iter(free_centers)
    free_iter_t = iter(free_thetas)
    for r in range(n_rungs):
        if r in anchor_idxs:
            idx = anchor_idxs.index(r)
            all_c.append(anchor_centers[idx])
            all_t.append(anchor_thetas[idx])
        else:
            all_c.append(next(free_iter_c))
            all_t.append(next(free_iter_t))
    return torch.stack(all_c), torch.stack(all_t)


def all_points_with_anchor(centers, thetas):
    coords = []
    for c, t in zip(centers, thetas):
        l, r = rung_endpoints(c, t)
        coords.append(l); coords.append(r)
    return torch.stack(coords)


# ------------------------
# Visualization
# ------------------------
def plot_system(coords, step):
    coords = coords.detach().cpu().numpy()
    plt.clf()
    # Draw sticks
    for r in range(n_rungs):
        l, rgt = coords[2*r], coords[2*r+1]
        plt.plot([l[0], rgt[0]], [l[1], rgt[1]], "k-", lw=2)
    # Draw springs
    for (r1, side1, r2, side2) in springs:
        i1 = 2*r1 + (0 if side1=="left" else 1)
        i2 = 2*r2 + (0 if side2=="left" else 1)
        p1, p2 = coords[i1], coords[i2]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "b--")
    # Points
    plt.scatter(coords[free_coord_idxs,0], coords[free_coord_idxs,1], c="blue")
    # Anchor rung
    plt.scatter(coords[anchor_coord_idxs,0], coords[anchor_coord_idxs,1], c="red", marker="s", s=100, label="anchor")
    plt.title(f"Ladder at step {step}")
    plt.axis("equal")
    plt.xlim(-1, 2)
    plt.ylim(-1, n_rungs*vertical_spacing)
    plt.pause(0.1)


# ------------------------
# Optimization
# ------------------------
plt.ion()
fig = plt.figure()

for step in range(1001):
    optimizer.zero_grad()

    centers_full, thetas_full = build_full_params(free_centers, free_thetas)
    coords = all_points_with_anchor(centers_full, thetas_full)

    # Compute spring energy
    energy = 0.0
    for (r1, side1, r2, side2), L0, k in zip(springs, spring_L, spring_k):
        i1 = 2*r1 + (0 if side1=="left" else 1)
        i2 = 2*r2 + (0 if side2=="left" else 1)
        dist = torch.norm(coords[i1] - coords[i2])
        energy += 0.5 * k * (dist - L0) ** 2

    energy.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}: Energy = {energy.item():.4f}")
        plot_system(coords, step)

plt.ioff()
plt.show()
