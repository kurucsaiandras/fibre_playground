import utils
import torch
import torch.nn.functional as F

def collision_loss(points, k, D, domain_size):
    boxes = utils.get_bounding_boxes(points, D*0.5)
    intersections, intersections_x_pbc, intersections_y_pbc, intersections_xy_pbc, intersections_yx_pbc = utils.get_bbox_intersections(boxes, domain_size)

    i_idx, j_idx = intersections.nonzero(as_tuple=True)
    dists = torch.norm(points[i_idx, :, None, :] - points[j_idx, None, :, :], dim=-1)

    # pbc x
    i_idx_x, j_idx_x = intersections_x_pbc.nonzero(as_tuple=True)
    dists_x = torch.norm((points[i_idx_x, :, None, :] - torch.tensor([domain_size[0],0,0], device=points.device)) - points[j_idx_x, None, :, :], dim=-1)
    # pbc y
    i_idx_y, j_idx_y = intersections_y_pbc.nonzero(as_tuple=True)
    dists_y = torch.norm((points[i_idx_y, :, None, :] - torch.tensor([0,domain_size[1],0], device=points.device)) - points[j_idx_y, None, :, :], dim=-1)
    # pbc xy
    i_idx_xy, j_idx_xy = intersections_xy_pbc.nonzero(as_tuple=True)
    dists_xy = torch.norm((points[i_idx_xy, :, None, :] - torch.tensor([domain_size[0],domain_size[1],0], device=points.device)) - points[j_idx_xy, None, :, :], dim=-1)
    # pbc yx
    i_idx_yx, j_idx_yx = intersections_yx_pbc.nonzero(as_tuple=True)
    dists_yx = torch.norm((points[i_idx_yx, :, None, :] - torch.tensor([domain_size[0],0,0], device=points.device)) - (points[j_idx_yx, None, :, :] - torch.tensor([0,domain_size[1],0], device=points.device)), dim=-1)
    # concatenate all distances
    dists = torch.cat([dists, dists_x, dists_y, dists_xy, dists_yx], dim=0)

    # penalty
    d_l = F.relu(D - dists)
    penalties = 0.5 * k * d_l*d_l

    loss = penalties.sum()
    return loss

def collision_loss_inf_wall(points, alpha, D):
    boxes = utils.get_bounding_boxes(points, D*0.5)
    intersections = utils.get_bbox_intersections(boxes)

    i_idx, j_idx = intersections.nonzero(as_tuple=True)

    dists = torch.norm(points[i_idx, :, None, :] - points[j_idx, None, :, :], dim=-1)

    # infinite wall penalty
    penalties = torch.clamp(-alpha * torch.log(dists / D), min=0.0)

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