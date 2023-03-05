import torch
import numpy as np
from private_transformers import PrivacyEngine
import math

class CustomizedPrivateEngine(PrivacyEngine):
    alpha: float = 0.5
    threshold = 0.85
    bits_noise_multiplier = 5.0
    lr_Z = 0.05
    max_search_iterations = 6
    last_max_norm = 0.5
    search_sigmas = [1, 1, 2, 2, 4, 4]
    
    def calc_clipped_cosine_sim(self, per_sample_norms, scale_factor):
        per_sample_clip_factor = (
            scale_factor / (per_sample_norms + 1e-6)
        ).clamp(max=1.0)

        dot_prod_sum = 0
        clipped_grad_squared_sum = 0
        true_grad_squared_sum = 0

        for _, param in self.named_params:
            clipped_grad = torch.flatten(torch.einsum("i,i...->...", per_sample_clip_factor, param.grad_sample))
            true_grad = torch.mean(torch.flatten(param.grad_sample, 1, -1), dim=0)
            clipped_grad_squared_sum += torch.sum(clipped_grad ** 2)
            true_grad_squared_sum += torch.sum(true_grad ** 2)
            dot_prod_sum += torch.sum(clipped_grad * true_grad)
        return (dot_prod_sum / (clipped_grad_squared_sum * true_grad_squared_sum).sqrt()).item()

    @torch.no_grad()
    def _accumulate_summed_grad(self, loss, scale):
        """Accumulate signal by summing clipped gradients.

        Removes `.grad_sample` and `.grad` for each variable that requires grad at the end.
        """
        with torch.enable_grad():
            loss.sum(dim=0).backward()

        norm_sample = []
        for name, param in self.named_params:
            try:
                batch_size = param.grad_sample.size(0)
            except AttributeError as error:
                args = error.args
                extra_msg = f"\n *** {name} parameter has no grad_sample attribute ***"
                error.args = (args[0] + extra_msg, *args[1:])
                raise error
            norm = param.grad_sample.reshape(batch_size, -1).norm(2, dim=1)
            norm_sample.append(norm)

        # The stack operation here is prone to error, thus clarify where the error is.
        per_sample_norms = torch.stack(norm_sample, dim=0).norm(2, dim=0)

        C = self.last_max_norm
        for _ in range(2):
            dt = (per_sample_norms > C).sum().item() / len(per_sample_norms) # percentage of samples in a batch that's bigger than the threshold * Z
            noisy_dt = dt + (torch.normal(0, self.bits_noise_multiplier, (1,)).item() * 1.0 / len(per_sample_norms))
            factor = math.exp(-self.lr_Z + noisy_dt)
            C = C * factor
        self.last_max_norm = C
        # print(f"Real_max: {per_sample_norms.max().item()}, C: {C}")
        upper = C
        lower = 1e-6
        t = 0
        while t < self.max_search_iterations and lower < upper:
            sigma = self.search_sigmas[t]
            mid = lower + (upper - lower) * self.alpha
            real_score = int(self.calc_clipped_cosine_sim(per_sample_norms, mid) > self.threshold)
            x = real_score + torch.normal(0, sigma, (1,)).item() * 1.0
            prob = math.exp(-(x/sigma) ** 2/2) / (math.exp(-(x/sigma) ** 2/2) + math.exp(-((x - 1)/sigma) ** 2/2))
            sampled_score = np.random.choice(2, p=[prob, 1-prob])
            if sampled_score == 1:
                upper = lower + (upper - lower) * self.alpha
            else:
                lower = lower + (upper - lower) * self.alpha
            t += 1

        scale_factor = lower + (upper - lower) * self.alpha
        scaled_per_param_norms = per_sample_norms / scale_factor
        per_sample_clip_factor = (
            self.max_grad_norm / (scaled_per_param_norms + 1e-6)
        ).clamp(max=1.0)

        for name, param in self.named_params:
            if not hasattr(param, 'summed_grad'):
                param.summed_grad = 0.
            current_device = param.grad_sample.device
            param.summed_grad += torch.einsum("i,i...->...", per_sample_clip_factor.to(current_device), param.grad_sample)

            # Aggressive memory saving -- delete everything except `.summed_grad` to save memory!
            if hasattr(param, "grad_sample"):
                # This must be deleted due to how `privacy_utils::supported_layers_grad_samplers.py` works!
                #   When a parameter with `.grad_sample` is reused, the per-sample gradients are accumulated!
                del param.grad_sample
            if hasattr(param, "grad"):
                del param.grad

        return norm_sample, per_sample_clip_factor