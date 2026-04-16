"""Flow Matching Euler Discrete Scheduler for Nucleus-Image."""

import mlx.core as mx


class FlowMatchEulerScheduler:
    def __init__(self, shift: float = 1.0, num_train_timesteps: int = 1000):
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps
        self.sigmas = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int):
        sigmas = mx.linspace(1.0, 0.0, num_inference_steps + 1)
        if self.shift != 1.0:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.sigmas = sigmas
        self.timesteps = sigmas[:-1] * self.num_train_timesteps

    def step(self, model_output, timestep_idx: int, sample):
        sigma = self.sigmas[timestep_idx]
        sigma_next = self.sigmas[timestep_idx + 1]
        dt = sigma_next - sigma
        return sample + dt * model_output
