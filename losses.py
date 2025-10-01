import utils
import torch
import torch.nn.functional as F

def collision_loss(points, k, D):
    boxes = utils.get_bounding_boxes(points, D*0.5)
    intersections = utils.get_bbox_intersections(boxes)

    i_idx, j_idx = intersections.nonzero(as_tuple=True)

    dists = torch.norm(points[i_idx, :, None, :] - points[j_idx, None, :, :], dim=-1)

    # penalty
    d_l = F.relu(D - dists)
    penalties = 0.5 * k * d_l*d_l

    loss = penalties.sum()
    return loss

def length_loss(points, k, L):
    """
    points: (n_fibres, resolution, 3)
    k: scalar or (n_fibres, ) tensor
    L: scalar or (n_fibres, ) tensor
    """
    diffs = points[:,:-1] - points[:,1:]      # (n_fibres, resolution-1, 3)
    dists = torch.norm(diffs, dim=2)    # (n_fibres, resolution-1)
    d_l = dists - L
    loss = 0.5 * k * d_l*d_l
    loss = loss.sum()
    return loss

def curvature_loss(points, k, L):
    """
    points: (n_fibres, resolution, 3)
    k: scalar or (n_fibres, ) tensor
    L: scalar or (n_fibres, ) tensor
    """
    p1 = points[:,:-2]       # (n_fibres, resolution-2, 3)
    p2 = points[:,1:-1]
    p3 = points[:,2:]
    angles = utils.angle_between(p1, p2, p3)  # (n_fibres, resolution-2)
    d_l = angles - L
    loss = 0.5 * k * d_l*d_l
    loss = loss.sum()
    return loss

def boundary_loss(points, domain_size, k):
    """
    points: (n_fibres, resolution, 3)
    domain_size: (3,) tensor/list specifying [x_max, y_max, z_max]
    k: scalar
    """
    # TODO THIS DOESNT TAKE INTO ACCOUNT THE FIBRE RADIUS
    # lower violations: values < 0
    lower_violation = torch.clamp(-points, min=0.0)
    # upper violations: values > domain_size
    upper_violation = torch.clamp(points - domain_size, min=0.0)
    # total violation per coordinate
    violations = lower_violation + upper_violation  # shape (n_fibres, resolution, 3)
    loss = 0.5 * k * (violations*violations).sum(dim=2)  # sum over x,y,z -> shape (n_fibres, resolution)
    loss = loss.sum()  # sum over all points
    return loss