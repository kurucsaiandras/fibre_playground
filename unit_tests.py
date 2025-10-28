import torch
import torch.nn.functional as F

def overlap_line_loss(fibre_coords, fibre_r, k_overlap, i_idx, j_idx):
    d_i = fibre_coords[i_idx, 1:, :] - fibre_coords[i_idx, :-1, :]  # (n_ij, res-1, 3)
    d_j = fibre_coords[j_idx, 1:, :] - fibre_coords[j_idx, :-1, :]  # (n_ij, res-1, 3)

    a = (d_i * d_i).sum(dim=2)  # (n_ij, res-1)
    e = (d_j * d_j).sum(dim=2)  # (n_ij, res-1)
    b = (d_i[:, :, None, :] * d_j[:, None, :, :]).sum(dim=3)  # (n_ij, res-1, res-1)

    r_ij = fibre_coords[i_idx, :-1, None, :] - fibre_coords[j_idx, None, :-1, :]  # (n_ij, res-1, res-1, 3)
    c = (d_i[:, :, None, :] * r_ij).sum(dim=-1)  # (n_ij, res-1, res-1)
    f = (d_j[:, :, None, :] * r_ij).sum(dim=-1)  # (n_ij, res-1, res-1)

    denom = (a * e)[:, :, None] - b * b  # (n_ij, res-1, res-1)
    denom_safe = denom.clone()
    denom_safe[denom_safe.abs() < 1e-8] = 1.0  # avoid division by 0 for parallel lines

    # Initial s and t
    s = torch.clamp((b * f - c * e[:, :, None]) / denom_safe, 0.0, 1.0)
    s[denom.abs() < 1e-8] = 0.0  # parallel lines
    t = (b * s + f) / e[:, :, None]

    # Clamp t and recompute s accordingly
    t_lt0 = t < 0.0
    t_gt1 = t > 1.0
    t = torch.clamp(t, 0.0, 1.0)

    # recompute s where t was clamped
    s_new_lt0 = torch.clamp(-c / a[:, :, None], 0.0, 1.0)
    s_new_gt1 = torch.clamp((b - c) / a[:, :, None], 0.0, 1.0)
    s = torch.where(t_lt0, s_new_lt0, s)
    s = torch.where(t_gt1, s_new_gt1, s)

    # --- at this point, s and t are correctly clamped and recomputed ---

    # Compute closest points (optional, depending on what loss uses)
    p_i = fibre_coords[i_idx, :-1, None, :] + s[..., None] * d_i[:, :, None, :]
    p_j = fibre_coords[j_idx, None, :-1, :] + t[..., None] * d_j[:, None, :, :]

    dists = torch.linalg.norm(p_i - p_j, dim=-1)
    print(dists)
    expected_dists = fibre_r[i_idx].view(-1, 1, 1) + fibre_r[j_idx].view(-1, 1, 1)

    # penalty
    d_l = F.relu(expected_dists - dists)
    penalties = 0.5 * k_overlap * d_l*d_l

    loss = penalties.sum()
    return loss

def main():
    # test function with a small example
    fibre_coords = torch.tensor([[[0.0, 0.0, 0.0],
                                 [1.0, 0.0, 0.0],
                                 [2.0, 0.0, 0.0]],
                                [[1.0, -0.5, 1.0],
                                 [1.0, 0.5, 1.0],
                                 [1.0, 1.5, 1.0]]])  # (2 fibres, 3 points, 3D)
    fibre_coords = torch.tensor([[[0.0, 0.0, 0.0],
                                 [1.0, 0.0, 0.0],
                                 [2.0, 0.0, 0.0]],
                                [[1.0, 0.0, 1.0],
                                 [2.0, 0.0, 1.0],
                                 [3.0, 0.0, 1.0]]])  # (2 fibres, 3 points, 3D)
    fibre_r = torch.tensor([0.6, 0.6])  # (2 fibres,)
    k_overlap = 10.0
    i_idx = torch.tensor([0])
    j_idx = torch.tensor([1])
    loss = overlap_line_loss(fibre_coords, fibre_r, k_overlap, i_idx, j_idx)
    print(f"Overlap line loss: {loss.item():.4f}")

    

if __name__ == "__main__":
    main()