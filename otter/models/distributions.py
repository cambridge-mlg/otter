from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch
import torch.distributions as dist
from typing_extensions import Self


class Distribution(ABC):
    num_stats: int

    def __init__(self) -> None:
        pass

    @abstractmethod
    def sample(self) -> torch.Tensor:
        pass

    @classmethod
    @abstractmethod
    def stack(cls, dists: List[Distribution], dim: int) -> Self:
        pass

    @abstractmethod
    def __add__(self, other: torch.Tensor) -> Distribution:
        pass

    @abstractmethod
    def __mul__(self, other: torch.Tensor) -> Distribution:
        pass


class Delta(Distribution):
    num_stats: int = 1

    def __init__(self, mean: torch.Tensor):
        self.mean = mean

    def sample(self) -> torch.Tensor:
        return self.mean

    @classmethod
    def stack(cls, dists: List[Distribution], dim: int) -> Self:
        mean = torch.stack([dist.mean for dist in dists], dim=dim)
        return cls(mean)

    def __add__(self, other: torch.Tensor) -> Delta:
        return Delta(self.mean + other)

    def __mul__(self, other: torch.Tensor) -> Delta:
        return Delta(self.mean * other)


def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


class Gaussian(Distribution):
    num_stats: int = 2

    def __init__(
        self,
        mean: torch.Tensor,
        unconstrained_stddev: torch.Tensor,
        min_std: float,
    ):
        self.mean = mean
        self.unconstrained_stddev = unconstrained_stddev
        self.stddev = torch.nn.functional.softplus(
            unconstrained_stddev
        )  # torch.ones_like(softplus(unconstrained_stddev) + eps)
        self.stddev = torch.clamp_min(self.stddev, min_std)
        self.dist = dist.Normal(self.mean, self.stddev)
        self.min_std = min_std

    def sample(self) -> torch.Tensor:
        return self.dist.rsample()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_prob: torch.Tensor = self.dist.log_prob(x)
        return log_prob

    @classmethod
    def stack(cls, dists: List[Distribution], dim: int) -> Self:
        mean = torch.stack([dist.mean for dist in dists], dim=dim)
        unconstrained_stddev = torch.stack(
            [dist.unconstrained_stddev for dist in dists], dim=dim
        )
        return cls(mean, unconstrained_stddev, dists[0].min_std)

    def __add__(self, other: torch.Tensor) -> Gaussian:
        return Gaussian(
            self.mean + other, self.unconstrained_stddev, self.min_std
        )

    def __mul__(self, other: torch.Tensor) -> Gaussian:
        unconstrained_stddev = inv_softplus(self.stddev * other)
        return Gaussian(self.mean * other, unconstrained_stddev, self.min_std)


def product_distribution(dists: List[Distribution], dim: int) -> Distribution:
    if not dists:
        raise ValueError("Cannot stack empty list of distributions")

    return dists[0].stack(dists, dim)
