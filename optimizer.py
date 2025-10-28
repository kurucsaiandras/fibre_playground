import torch

class Optimizer:
    def __init__(self, config, rve):
        self.config = config
        self.losses = {}
        self.loss_sum = 0
        self.phase_iter = 0
        self.rve = rve
        self._init_joint()
        if config.alternate_phases:
            self._init_overlap()
            self.current = self.overlap
            self.phase = 'overlap'
            self.avg_window = config.moving_average_window
            self.moving_avg = torch.zeros(self.avg_window*2, device=rve.device)
            self.min_loss = torch.inf
            self.params_checkpoint = rve.fibre_coords.clone()
        else:
            self.current = self.joint
            self.phase = 'joint'
    
    def _init_joint(self):
        if self.config.joint_optimizer == 'adam':
            self.joint = torch.optim.Adam([self.rve.fibre_coords], lr=self.config.learning_rate)
        elif self.config.joint_optimizer == 'lbfgs':
            self.joint = torch.optim.LBFGS([self.rve.fibre_coords], lr=self.config.learning_rate,
                                               max_iter=10, history_size=10)
            
    def _init_overlap(self):
        self.overlap = torch.optim.SGD([self.rve.fibre_coords], lr=1e-3)

    def loss(self):
        if self.config.line_loss:
            self.losses["overlap"] = self.rve.overlap_line_loss(self.phase)
        else:
            self.losses["overlap"] = self.rve.overlap_loss()
        self.losses["boundary"] = self.rve.boundary_loss()
        self.losses["length"] = self.rve.length_loss()
        self.losses["curvature"] = self.rve.curvature_loss()
        if self.phase == 'joint': self.loss_sum = sum(self.losses.values())
        elif self.phase == 'overlap': self.loss_sum = self.losses["overlap"]

    def switch_phase(self):
        if self.phase == 'overlap':
            print(f"Took {self.phase_iter} iterations in 'overlap' phase, switching to 'joint'")
            self.phase = 'joint'
            self.moving_avg = torch.zeros(self.avg_window*2, device=self.rve.device)
            self.min_loss = torch.inf
            if self.config.reset_optimizers:
                self.current - self._init_joint()
            else:
                self.current = self.joint
        elif self.phase == 'joint':
            print(f"Took {self.phase_iter} iterations in 'joint' phase, switching to 'overlap'")
            self.phase = 'overlap'
            if self.config.reset_optimizers:
                self.current = self._init_overlap()
            else:
                self.current = self.overlap
        self.phase_iter = 0

    def save_checkpoint(self):
        '''
        If the current loss is smaller than the previous minimum,
        updates min loss and saves current model params.
        '''
        if self.loss_sum < self.min_loss:
            self.min_loss = self.loss_sum
            self.params_checkpoint = self.rve.fibre_coords.clone()

    def update_moving_avg(self):
        if self.phase_iter < self.avg_window*2:
            self.moving_avg[self.phase_iter] = self.loss_sum
        else:
            self.moving_avg = torch.roll(self.moving_avg, -1)
            self.moving_avg[-1] = self.loss_sum

    def eval_joint_criteria(self):
        if self.phase == 'joint':
            return (self.loss_sum < self.config.cumulative_loss_threshold
                    or self.phase_iter == self.config.max_iter_per_phase
                    or self.moving_avg[:self.avg_window].sum() < self.moving_avg[self.avg_window:].sum())
        else: raise Exception(f"Tried to evaluate joint criteria in phase '{self.phase}'")
        
    def lbfgs_closure(self):
        self.current.zero_grad()
        losses = {}
        if self.config.line_loss:
            losses["overlap"] = self.rve.overlap_line_loss(self.phase)
        else:
            losses["overlap"] = self.rve.overlap_loss()
        losses["boundary"] = self.rve.boundary_loss()
        losses["length"] = self.rve.length_loss()
        losses["curvature"] = self.rve.curvature_loss()
        # assuming lbfgs is the joint optimizer
        loss_sum = sum(losses.values())
        loss_sum.backward()
        return loss_sum

    def step(self):
        if not torch.isfinite(self.loss_sum):
            raise Exception("Non-finite loss in closure -> aborting")
        if self.phase == 'joint' and self.config.joint_optimizer == 'lbfgs':
            self.current.step(self.lbfgs_closure)
        else:
            self.current.zero_grad()
            self.loss_sum.backward()
            if self.config.grad_clipping:
                torch.nn.utils.clip_grad_norm_([self.rve.fibre_coords],
                                            self.config.max_grad_norm)
            self.current.step()
    
    def project_gradients(self):
        grads = {}
        for name, loss in self.losses.items():
            self.current.zero_grad()
            loss.backward(retain_graph=True)
            grads[name] = self.rve.fibre_coords.grad.clone()

        # Apply projection: ensure overlap gradient is not increased
        g_total = sum(grads)
        # element-wise dot product and norm
        dot = (g_total * grads['overlap']).sum(dim=-1, keepdim=True)
        g_proj = dot / (grads['overlap'] * grads['overlap']).sum(dim=-1, keepdim=True) * grads['overlap']
        g_total = torch.where(dot < 0, g_total - g_proj, g_total)

        # Set projected gradients and step
        with torch.no_grad():
            self.rve.fibre_coords.grad = g_total