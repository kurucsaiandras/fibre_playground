import numpy as np
import matplotlib.pyplot as plt

def get_cumulative_force(circles, r, gap, idx):
    """
    Calculate the cumulative force on a circle due to repulsive forces from other circles.
    
    Parameters:
    circles (np array): An array of shape (N, 2) representing the positions of N circles.
    r (float): The radius of each circle.
    gap (float): The minimum allowed distance between circles.
    idx (int): The index of the circle for which to calculate the force.
    """
    force = np.array([0.0, 0.0])
    pos = circles[idx]
    for i, other_pos in enumerate(circles):
        if i != idx:
            direction = pos - other_pos
            distance = np.linalg.norm(direction)
            if distance < 2 * r + gap and distance > 1e-5:  # Avoid division by zero
                direction /= distance  # Normalize
                overlap = 2 * r + gap - distance
                force += direction * overlap
    return force

def update_positions(circles, r, gap, step_size):
    """
    Update the positions of circles based on cumulative forces.
    
    Parameters:
    circles (np array): An array of shape (N, 2) representing the positions of N circles.
    r (float): The radius of each circle.
    gap (float): The minimum allowed distance between circles.
    step_size (float): The step size for position updates.
    """
    new_circles = circles.copy()
    for i in range(len(circles)):
        force = get_cumulative_force(circles, r, gap, i)
        new_circles[i] += force * step_size
    return new_circles

def plot_circles(ax, circles, r):
    """
    Plot the circles using matplotlib.
    
    Parameters:
    ax (matplotlib axis): The axis on which to plot the circles.
    circles (np array): An array of shape (N, 2) representing the positions of N circles.
    r (float): The radius of each circle.
    """
    ax.clear()
    for pos in circles:
        circle = plt.Circle(pos, r, fill=False)
        ax.add_artist(circle)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal', 'box')
    ax.grid()

num_circles = 100
radius = 0.5
gap = 0.01
positions = np.random.rand(num_circles, 2) * 10  # Random initial positions
num_iterations = 100
step_size = 0.5

plt.ion()  # interactive mode ON
fig, ax = plt.subplots()

for _ in range(num_iterations):
    positions = update_positions(positions, radius, gap, step_size)
    plot_circles(ax, positions, radius)
    plt.pause(0.05)

plt.ioff()  # turn off interactive mode
plt.show()