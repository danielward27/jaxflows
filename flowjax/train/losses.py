"""Common loss functions for training normalizing flows.

In order to be compatible with ``fit_to_data``, the loss function arguments must match
``(params, static, x, condition, key)``, where ``params`` and ``static`` are the
partitioned model (see ``equinox.partition``).

For ``fit_to_key_based_loss``, the loss function signature must match
``(params, static, key)``.
"""

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import paramax
from jax import vmap
from jax.lax import stop_gradient
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray

from flowjax.distributions import AbstractDistribution


class MaximumLikelihoodLoss:
    """Loss for fitting a flow with maximum likelihood (negative log likelihood).

    This loss can be used to learn either conditional or unconditional distributions.
    """

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractDistribution,
        static: AbstractDistribution,
        x: Array,
        condition: Array | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, ""]:
        """Compute the loss. Key is ignored (for consistency of API)."""
        dist = paramax.unwrap(eqx.combine(params, static))
        return -dist.log_prob(x, condition).mean()


class WeightedMaximumLikelihoodLoss:
    """Loss for fitting a flow with maximum likelihood (negative log likelihood)
    for weighted samples.

    This loss can be used to learn either conditional or unconditional distributions.
    """
    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractDistribution,
        static: AbstractDistribution,
        x: Array,
        weights: Array, 
        condition: Array | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, ""]:
        """Compute the loss."""
        dist = paramax.unwrap(eqx.combine(params, static))
        evl = -dist.log_prob(x, condition)
        return (evl*weights).sum()/len(evl)


class ContrastiveLoss:
    r"""Loss function for use in a sequential neural posterior estimation algorithm.

    Learns a posterior ``p(x|condition)``. Contrastive samples for each ``x`` are
    generated from other x samples in the batch.

    Note, that in a simulation based inference context, often :math:`x` is used to
    denote simulations, and :math:`\theta` for simulation parameters. However, for
    consistency with the rest of the package, we use ``x`` to represent the target
    variable (the simulator parameters), and ``condition`` for the conditioning variable
    (the simulator output/oberved data).

    Args:
        prior: The prior distribution over x (the target
            variable).
        n_contrastive: The number of contrastive samples/atoms to use when
            computing the loss. Must be less than ``batch_size``.

    References:
        - https://arxiv.org/abs/1905.07488
        - https://arxiv.org/abs/2002.03712
    """

    def __init__(self, prior: AbstractDistribution, n_contrastive: int):
        self.prior = prior
        self.n_contrastive = n_contrastive

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractDistribution,
        static: AbstractDistribution,
        x: Float[Array, "..."],
        condition: Array | None,
        key: PRNGKeyArray,
    ) -> Float[Array, ""]:
        """Compute the loss."""
        if x.shape[0] <= self.n_contrastive:
            raise ValueError(
                f"Number of contrastive samples {self.n_contrastive} must be less than "
                f"the size of x {x.shape}.",
            )

        dist = paramax.unwrap(eqx.combine(params, static))

        def single_x_loss(x_i, condition_i, contrastive_idxs):
            positive_logit = dist.log_prob(x_i, condition_i) - self.prior.log_prob(x_i)
            contrastive = x[contrastive_idxs]
            contrastive_logits = dist.log_prob(
                contrastive, condition_i
            ) - self.prior.log_prob(contrastive)
            normalizer = logsumexp(jnp.append(contrastive_logits, positive_logit))
            return -(positive_logit - normalizer)

        contrastive_idxs = _get_contrastive_idxs(key, x.shape[0], self.n_contrastive)
        return eqx.filter_vmap(single_x_loss)(x, condition, contrastive_idxs).mean()


def _get_contrastive_idxs(key: PRNGKeyArray, batch_size: int, n_contrastive: int):

    @eqx.filter_vmap
    def _get_idxs(key, idx, batch_size, n_contrastive):
        choices = jnp.delete(jnp.arange(batch_size), idx, assume_unique_indices=True)
        return jr.choice(key, choices, (n_contrastive,), replace=False)

    keys = jr.split(key, batch_size)
    return _get_idxs(keys, jnp.arange(batch_size), batch_size, n_contrastive)


class ElboLoss:
    """The negative evidence lower bound (ELBO), approximated using samples.

    Args:
        num_samples: Number of samples to use in the ELBO approximation.
        target: The target, i.e. log posterior density up to an additive constant / the
            negative of the potential function, evaluated for a single point.
        stick_the_landing: Whether to use the (often) lower variance ELBO gradient
            estimator introduced in https://arxiv.org/pdf/1703.09194.pdf. Note for flows
            this requires evaluating the flow in both directions (running the forward
            and inverse transformation). For some flow architectures, this may be
            computationally expensive due to assymetrical computational complexity
            between the forward and inverse transformation. Defaults to False.
    """

    target: Callable[[ArrayLike], Array]
    num_samples: int
    stick_the_landing: bool

    def __init__(
        self,
        target: Callable[[ArrayLike], Array],
        num_samples: int,
        *,
        stick_the_landing: bool = False,
    ):
        self.target = target
        self.num_samples = num_samples
        self.stick_the_landing = stick_the_landing

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractDistribution,
        static: AbstractDistribution,
        key: PRNGKeyArray,
    ) -> Float[Array, ""]:
        """Compute the ELBO loss.

        Args:
            params: The trainable parameters of the model.
            static: The static components of the model.
            key: Jax random key.
        """
        dist = eqx.combine(params, static)

        if self.stick_the_landing:
            # Requries both forward and inverse pass
            samples = dist.sample(key, (self.num_samples,))
            dist = eqx.combine(stop_gradient(params), static)
            log_probs = dist.log_prob(samples)

        else:
            # Requires only forward pass through the flow.
            samples, log_probs = dist.sample_and_log_prob(key, (self.num_samples,))

        target_density = vmap(self.target)(samples)
        return (log_probs - target_density).mean()
