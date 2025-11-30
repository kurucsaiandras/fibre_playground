import torch
import torch.nn.functional as F
import utils
import math
import os
import mem

class RVE:
    def __init__(self, config, device):
        # set up physical constants
        self.k_length = config.spring_system.k_length
        self.k_curvature = config.spring_system.k_curvature
        self.k_boundary = config.spring_system.k_boundary
        self.k_overlap = config.spring_system.k_overlap

        self.device = device
        self.apply_pbc = config.evolution.apply_pbc
        if config.initialization.method == 'generate':
            self.domain_size = torch.tensor(
                config.initialization.generate.domain_size_initial, device=device)
            if config.initialization.generate.method == 'poisson':
                self.fibre_coords, self.l0_length, self.phi0_curvature = utils.generate_fibres_poisson(config, device)
            elif config.initialization.generate.method == 'random':
                self.fibre_coords, self.l0_length, self.phi0_curvature = utils.generate_fibres_random(config, device)
            elif config.initialization.generate.method == 'curl':
                self.fibre_coords, self.l0_length, self.phi0_curvature = utils.generate_fibres_curl(config, device)
            self.fibre_r, self.fibre_r_target = utils.generate_radii(self.fibre_coords.shape[0], config, device)
        elif config.initialization.method == 'load':
            self.load(config.initialization.load.name, config.initialization.load.step)
            if config.evolution.apply_pbc != self.apply_pbc:
                print("Warning: loaded RVE has different PBC setting than current config.")
                print(f"Using setting from config: apply_pbc = {config.evolution.apply_pbc}")
                self.apply_pbc = config.evolution.apply_pbc
        # make fibre_coords torch parameter
        self.fibre_coords = torch.nn.Parameter(self.fibre_coords)
        # calculate r step
        self.r_incr = (self.fibre_r_target - self.fibre_r) / config.evolution.fibre_r_steps
        # calculate domain size step
        self.domain_size_target = torch.tensor(config.evolution.domain_size_target, device=device)
        self.domain_size_incr = (self.domain_size_target - self.domain_size) / config.evolution.domain_size_steps
        
    @classmethod
    def eval(cls, name, step, device):
        """Alternative constructor for evaluation-only mode."""
        self = cls.__new__(cls)  # create instance without calling __init__
        self.device = device
        self.load(name, step)
        self.fibre_coords = self.fibre_coords.detach()  # to use in eval mode
        return self
    
    @classmethod
    def external(cls, fibre_coords, radius, domain_size, downsample):
        """Alternative constructor for 3rd party data that only has fibre coordinates."""
        self = cls.__new__(cls)  # create instance without calling __init__
        self.fibre_coords = fibre_coords
        if downsample:
            self.fibre_coords = self.fibre_coords[:, ::20, :] # assuming apx 900 points, and we want apx 40
        self.l0_length = self.fibre_coords[:,:-1] - self.fibre_coords[:,1:]      # (n_fibres, resolution-1, 3)
        if domain_size is None:
            self.domain_size = torch.tensor([fibre_coords[:,:,0].max(), fibre_coords[:,:,1].max(), fibre_coords[:,:,2].max()], device=fibre_coords.device)
        else:
            self.domain_size = domain_size
        self.fibre_r = torch.full((fibre_coords.shape[0],), radius, device=fibre_coords.device)
        self.fibre_r_target = self.fibre_r.clone()
        self.apply_pbc = False
        return self

    def load(self, name, step):
        path = f"results/{name}/rve/{step}.pt"
        state = torch.load(path, map_location=self.device)
        self.fibre_coords = state['fibre_coords']
        self.l0_length = state['l0_length']
        self.phi0_curvature = state['phi0_curvature']
        self.domain_size = state['domain_size']
        self.fibre_r = state['fibre_r']
        self.fibre_r_target = state['fibre_r_target']
        self.apply_pbc = state['apply_pbc']

    def save(self, job_name, step, time):
        save_dict = {
            "fibre_coords": self.fibre_coords,
            "l0_length": self.l0_length,
            "phi0_curvature": self.phi0_curvature,
            "time": time,
            "step": step,
            "domain_size": self.domain_size,
            "fibre_r": self.fibre_r,
            "fibre_r_target": self.fibre_r_target,
            "apply_pbc": self.apply_pbc
        }
        os.makedirs(f"results/{job_name}/rve", exist_ok=True)
        torch.save(save_dict, f"results/{job_name}/rve/{step}.pt")

    def save_unit_test(self, job_name, step, time):
        fibre_coords = torch.tensor([[[0.4, 0.5, 0.0],
                                      [0.4, 0.5, 0.25],
                                      [0.4, 0.5, 0.5],
                                      [0.4, 0.5, 0.75],
                                      [0.4, 0.5, 1.0]],
                                     [[0.6, 0.5, 0.0],
                                      [0.6, 0.5, 0.25],
                                      [0.6, 0.5, 0.5],
                                      [0.6, 0.5, 0.75],
                                      [0.6, 0.5, 1.0]]], device=self.device)
        l0_length = torch.tensor([[0.25], [0.25]], device=self.device)
        fibre_r = torch.tensor([0.15, 0.15], device=self.device)
        fibre_r_target = torch.tensor([0.15, 0.15], device=self.device)
        save_dict = {
            "fibre_coords": fibre_coords,
            "l0_length": l0_length,
            "time": time,
            "step": step,
            "domain_size": self.domain_size,
            "fibre_r": fibre_r,
            "fibre_r_target": fibre_r_target,
            "apply_pbc": self.apply_pbc
        }
        torch.save(save_dict, f"results/{job_name}/rve/{step}.pt")

    def evolve(self):
        """
        Evolve the RVE by one step in fibre radii or domain size.
        Returns True if evolution is complete (both radii and domain size reached target).
        """
        # first, increase fibre radii until target (account for float precision)
        if torch.any(self.fibre_r < self.fibre_r_target - 1e-6):
            self.fibre_r = self.fibre_r + self.r_incr
            return False
        # then, decrease domain size until target (account for float precision)
        if torch.any(self.domain_size > self.domain_size_target + 1e-4):
            proportion = (self.domain_size + self.domain_size_incr) / self.domain_size
            self.domain_size *= proportion
            self.fibre_coords.data[:, :, 0] *= proportion[0]
            self.fibre_coords.data[:, :, 1] *= proportion[1]
            return False
        return True

    def get_fibre_to_volume_ratio(self):
        return torch.pow(self.fibre_r, 2).sum() * math.pi / (self.domain_size[0] * self.domain_size[1])
    
    def resolve_overlaps_step_own(self):
        boxes = utils.get_bounding_boxes(self.fibre_coords, self.fibre_r)
        intersections = utils.get_bbox_intersections(boxes, self.domain_size, self.apply_pbc)

        i_idx, j_idx = intersections["normal"].nonzero(as_tuple=True)

        d_i = self.fibre_coords[i_idx, 1:, :] - self.fibre_coords[i_idx, :-1, :]  # (n_ij, res-1, 3)
        d_j = self.fibre_coords[j_idx, 1:, :] - self.fibre_coords[j_idx, :-1, :]  # (n_ij, res-1, 3)

        a = (d_i * d_i).sum(dim=2)  # (n_ij, res-1)
        e = (d_j * d_j).sum(dim=2)  # (n_ij, res-1)
        b = (d_i[:, :, None, :] * d_j[:, None, :, :]).sum(dim=3)  # (n_ij, res-1, res-1)

        r_ij = self.fibre_coords[i_idx, :-1, None, :] - self.fibre_coords[j_idx, None, :-1, :]  # (n_ij, res-1, res-1, 3)
        c = (d_i[:, :, None, :] * r_ij).sum(dim=-1)  # (n_ij, res-1, res-1)
        f = (d_j[:, :, None, :] * r_ij).sum(dim=-1)  # (n_ij, res-1, res-1)

        denom = (a * e)[:, :, None] - b * b  # (n_ij, res-1, res-1)
        denom_safe = denom.clone()
        denom_safe[denom_safe.abs() < 1e-8] = 1.0  # avoid division by 0 for parallel lines

        # Initial s and t
        s = torch.clamp((b * f - c * e[:, :, None]) / denom_safe, 0.0, 1.0)
        s[denom.abs() < 1e-8] = 0.5  # parallel lines
        t = (b * s + f) / e[:, :, None]

        # Clamp t and recompute s accordingly
        t_lt0 = t < 0.0
        t_gt1 = t > 1.0
        t = torch.clamp(t, 0.0, 1.0)  # (n_ij, res-1, res-1)

        # recompute s where t was clamped
        s_new_lt0 = torch.clamp(-c / a[:, :, None], 0.0, 1.0)
        s_new_gt1 = torch.clamp((b - c) / a[:, :, None], 0.0, 1.0)
        s = torch.where(t_lt0, s_new_lt0, s)
        s = torch.where(t_gt1, s_new_gt1, s)  # (n_ij, res-1, res-1)

        p_i = self.fibre_coords[i_idx, :-1, None, :] + s[..., None] * d_i[:, :, None, :]
        p_j = self.fibre_coords[j_idx, None, :-1, :] + t[..., None] * d_j[:, None, :, :]
        i_to_j = p_j - p_i
        real_dists = torch.linalg.norm(p_i - p_j, dim=-1)  # (n_ij, res-1)
        min_expected_dists = self.fibre_r[i_idx].view(-1, 1, 1) + self.fibre_r[j_idx].view(-1, 1, 1)  # (n_ij, res-1, res-1)
        diff = min_expected_dists - real_dists
        diff = torch.clamp(diff, min=0)  # (n_ij, res-1, res-1)
        #to_update = torch.where(diff > 0)
        penalties = 0.5 * self.k_overlap * diff*diff
        loss = penalties.sum()

        diff_vec = i_to_j / torch.norm(i_to_j, dim=-1, keepdim=True) * diff[..., None]  # (n_ij, res-1, res-1, 3)
        
        # INDICES I
        to_add_i = -self.fibre_r[j_idx].view(-1, 1, 1, 1) / min_expected_dists[..., None] * diff_vec   # (n_ij, res-1, res-1, 3)
        # p1 gets a contribution if s is not 1.0 (meaning the shortest distance is completely at p2 point)
        corr_i_p1 = torch.nn.functional.pad((to_add_i * torch.ceil(1.0 - s[..., None])), (0, 0, 0, 0, 0, 1))  # (n_ij, res, res-1, 3)
        # p2 gets a contribution if s is not 0.0 (meaning the shortest distance is completely at p1 point)
        corr_i_p2 = torch.nn.functional.pad((to_add_i * torch.ceil(s[..., None])), (0, 0, 0, 0, 1, 0))  # (n_ij, res, res-1, 3)
        corr_i = torch.cat([corr_i_p1, corr_i_p2], dim=2)
        mask_i = (corr_i != 0).any(dim=-1, keepdim=True)  # shape: (n_ij, res, 2*(res-1), 1)
        sum_nonzero_i = (corr_i * mask_i).sum(dim=2)  # shape: (n_ij, res, 3)
        count_nonzero_i = mask_i.sum(dim=2).clamp(min=1)  # shape: (n_ij, res, 1)
        corr_i_final = sum_nonzero_i / count_nonzero_i  # shape: (n_ij, res, 3)

        # INDICES J
        to_add_j = self.fibre_r[i_idx].view(-1, 1, 1, 1) / min_expected_dists[..., None] * diff_vec    # (n_ij, res-1, res-1, 3)
        corr_j_p1 = torch.nn.functional.pad((to_add_j * torch.ceil(1.0 - t[..., None])), (0, 0, 0, 1, 0, 0))  # (n_ij, res-1, res, 3)
        corr_j_p2 = torch.nn.functional.pad((to_add_j * torch.ceil(t[..., None])), (0, 0, 1, 0, 0, 0))  # (n_ij, res-1, res, 3)
        corr_j = torch.cat([corr_j_p1, corr_j_p2], dim=1)
        mask_j = (corr_j != 0).any(dim=-1, keepdim=True)  # shape: (n_ij, 2*(res-1), res, 1)
        sum_nonzero_j = (corr_j * mask_j).sum(dim=1)  # shape: (n_ij, res, 3)
        count_nonzero_j = mask_j.sum(dim=1).clamp(min=1)  # shape: (n_ij, res, 1)
        corr_j_final = sum_nonzero_j / count_nonzero_j  # shape: (n_ij, res, 3)

        idx = torch.cat([i_idx, j_idx])  # (n_ij*2)
        to_add = torch.cat([corr_i_final, corr_j_final])  # (n_ij*2, res, 3)

        mask = (to_add.norm(dim=-1) > 0).float()   # (n_ij*2, res)

        n_fibres, res, _ = self.fibre_coords.shape

        with torch.no_grad():
            # --- accumulate vector sums ---
            sum_buf = torch.zeros_like(self.fibre_coords)
            sum_buf.scatter_add_(
                0,
                idx[:, None, None].expand(-1, res, 3),
                to_add
            )

            # --- accumulate nonzero counts (for averaging) ---
            count_buf = torch.zeros(n_fibres, res, 1, device=self.fibre_coords.device)
            count_buf.scatter_add_(
                0,
                idx[:, None, None].expand(-1, res, 1),
                mask[..., None]   # shape (n_ij*2, res, 1)
            )

            # --- compute averages safely (avoid division by zero) ---
            avg_buf = torch.where(
                count_buf > 0,
                sum_buf / count_buf,
                torch.zeros_like(sum_buf)
            )

            # --- apply averaged correction ---
            self.fibre_coords += avg_buf
        return loss

    def overlap_line_loss(self, phase, batch_size=100000): #50000
        #mem.print_cuda_mem("before fun")
        boxes = utils.get_bounding_boxes(self.fibre_coords, self.fibre_r)
        intersections = utils.get_bbox_intersections(boxes, self.domain_size, self.apply_pbc)

        offs = utils.get_pbc_offsets(self.domain_size, self.device)
        cases = ["normal"]
        if self.apply_pbc:
            cases = ["normal", "x_pbc", "y_pbc", "xy_pbc", "yx_pbc"]
        total_loss = 0.0

        for case in cases:
            i_idxs, j_idxs = intersections[case].nonzero(as_tuple=True)
            n_pairs = i_idxs.shape[0]

            for start in range(0, n_pairs, batch_size):
                end = min(start + batch_size, n_pairs)
                i_idx = i_idxs[start:end]
                j_idx = j_idxs[start:end]
                d_i = self.fibre_coords[i_idx, 1:, :] - self.fibre_coords[i_idx, :-1, :]  # (n_ij, res-1, 3)
                d_j = self.fibre_coords[j_idx, 1:, :] - self.fibre_coords[j_idx, :-1, :]  # (n_ij, res-1, 3)

                a = (d_i * d_i).sum(dim=2)  # (n_ij, res-1)
                e = (d_j * d_j).sum(dim=2)  # (n_ij, res-1)
                b = (d_i[:, :, None, :] * d_j[:, None, :, :]).sum(dim=3)  # (n_ij, res-1, res-1)

                r_ij = (self.fibre_coords[i_idx, :-1, None, :]-offs[case]['i']) - (self.fibre_coords[j_idx, None, :-1, :]-offs[case]['j'])  # (n_ij, res-1, res-1, 3)
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

                p_i = (self.fibre_coords[i_idx, :-1, None, :]-offs[case]['i']) + s[..., None] * d_i[:, :, None, :]
                p_j = (self.fibre_coords[j_idx, None, :-1, :]-offs[case]['j']) + t[..., None] * d_j[:, None, :, :]

                dists = torch.linalg.norm(p_i - p_j, dim=-1)
                expected_dists = self.fibre_r[i_idx].view(-1, 1, 1) + self.fibre_r[j_idx].view(-1, 1, 1)

                # penalty
                d_l = F.relu(expected_dists - dists, inplace=True)
                if phase == 'joint':
                    penalties = 0.5 * self.k_overlap * d_l*d_l
                    batch_loss = penalties.sum()
                elif phase == 'overlap':
                    batch_loss = d_l.sum()
                batch_loss.backward()
                total_loss += float(batch_loss.detach())
                #mem.print_cuda_mem("in batch before del")
                # free memory for this batch
                del d_i, d_j, a, e, b, r_ij, c, f, denom, denom_safe
                del s, t, s_new_lt0, s_new_gt1, p_i, p_j, dists, expected_dists, d_l
                if phase == 'joint':
                    del penalties
                torch.cuda.empty_cache()
                #mem.print_cuda_mem("in batch after del")

        #mem.print_cuda_mem("after fun")
        return total_loss

    def overlap_loss(self):
        boxes = utils.get_bounding_boxes(self.fibre_coords, self.fibre_r)
        intersections = utils.get_bbox_intersections(boxes, self.domain_size, self.apply_pbc)

        offs = utils.get_pbc_offsets(self.domain_size, self.device)
        cases = ["normal"]
        if self.apply_pbc:
            cases = ["normal", "x_pbc", "y_pbc", "xy_pbc", "yx_pbc"]
        total_loss = 0.0

        for case in cases:
            i_idx, j_idx = intersections[case].nonzero(as_tuple=True)
            dists = torch.norm((self.fibre_coords[i_idx, :, None, :]-offs[case]['i'])
                                -(self.fibre_coords[j_idx, None, :, :]-offs[case]['j']), dim=-1)
            expected_dists = self.fibre_r[i_idx].view(-1,1,1) + self.fibre_r[j_idx].view(-1,1,1)

            # penalty
            d_l = F.relu(expected_dists - dists, inplace=True)
            
            penalties = 0.5 * self.k_overlap * d_l*d_l
            case_loss = penalties.sum()
            case_loss.backward()
            total_loss += float(case_loss.detach())
        return total_loss

    def length_loss(self, no_grad=False):
        """
        self.fibre_coords: (n_fibres, resolution, 3)
        self.k_length: scalar or (n_fibres, ) tensor
        self.l0_length: scalar or (n_fibres, ) tensor
        """
        diffs = self.fibre_coords[:,:-1] - self.fibre_coords[:,1:]      # (n_fibres, resolution-1, 3)
        dists = torch.norm(diffs, dim=2)    # (n_fibres, resolution-1)
        d_l = dists - self.l0_length
        loss = 0.5 * self.k_length * d_l*d_l
        loss = loss.sum()
        if not no_grad:
            loss.backward()
        return float(loss.detach())
    
    def equal_segments_loss(self, no_grad=False):
        segment_lengths = (self.fibre_coords[:,:-1] - self.fibre_coords[:,1:]).norm(dim=2)  # (n_fibres, resolution-1)
        d_l = segment_lengths - segment_lengths.mean(dim=-1)
        loss = 0.5 * self.k_length * d_l*d_l
        loss = loss.sum()
        if not no_grad:
            loss.backward()
        return float(loss.detach())

    def curvature_loss(self, no_grad=False):
        """
        self.fibre_coords: (n_fibres, resolution, 3)
        self.k_curvature: scalar
        self.phi0_curvature: scalar
        """
        p1 = self.fibre_coords[:,:-2]       # (n_fibres, resolution-2, 3)
        p2 = self.fibre_coords[:,1:-1]
        p3 = self.fibre_coords[:,2:]
        angles = utils.angle_between(p1, p2, p3)  # (n_fibres, resolution-2)
        d_l = angles - self.phi0_curvature
        loss = 0.5 * self.k_curvature * d_l*d_l
        loss = loss.sum()
        if not no_grad:
            loss.backward()
        return float(loss.detach())

    def boundary_loss(self, no_grad=False):
        """
        self.fibre_coords: (n_fibres, resolution, 3)
        self.fibre_r: (n_fibres,) tensor of fibre radii
        self.domain_size: (3,) tensor/list specifying [x_max, y_max, z_max]
        self.k_boundary: scalar
        """
        if self.apply_pbc:
            # lower violations: values < 0
            lower_violation = torch.clamp(-self.fibre_coords, min=0.0)
            # upper violations: values > domain_size
            upper_violation = torch.clamp(self.fibre_coords - self.domain_size, min=0.0)
        else:
            # fibre radii matter only in x and y directions
            fibre_r_offset = torch.stack([
                self.fibre_r,  # x
                self.fibre_r,  # y
                torch.zeros_like(self.fibre_r)  # z
            ], dim=1).unsqueeze(1)  # (n_fibres, 1, 3)
            # lower violations: values < fibre_r
            lower_violation = torch.clamp(fibre_r_offset - self.fibre_coords, min=0.0)
            # upper violations: values > domain_size - fibre_r
            upper_violation = torch.clamp(self.fibre_coords - (self.domain_size - fibre_r_offset), min=0.0)
        # total violation per coordinate
        violations = lower_violation + upper_violation  # shape (n_fibres, resolution, 3)
        loss = 0.5 * self.k_boundary * (violations*violations).sum(dim=2)  # sum over x,y,z -> shape (n_fibres, resolution)
        loss = loss.sum()  # sum over all points
        if not no_grad:
            loss.backward()
        return float(loss.detach())
    
    def snap_z_to_surface_loss(self, no_grad=False):
        lower_violation = torch.clamp(self.fibre_coords[:, 0, 2], min=0.0) # violate if z > 0
        upper_violation = torch.clamp(self.fibre_coords[:, -1, 2], max=self.domain_size[2]) # violate if z < domain_size
        violations = lower_violation + upper_violation  # shape (n_fibres,)
        loss = 0.5 * self.k_boundary * (violations*violations).sum()
        loss = loss.sum()  # sum over all points
        if not no_grad:
            loss.backward()
        return float(loss.detach())