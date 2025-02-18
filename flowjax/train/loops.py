"""Training loops."""

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
import paramax
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree, Scalar
from tqdm import tqdm

from flowjax.train.losses import MaximumLikelihoodLoss
from flowjax.train.losses import WeightedMaximumLikelihoodLoss
from flowjax.train.train_utils import (
    count_fruitless,
    get_batches,
    step,
    train_val_split,
)


def fit_to_key_based_loss(
    key: PRNGKeyArray,
    tree: PyTree,
    *,
    loss_fn: Callable[[PyTree, PyTree, PRNGKeyArray], Scalar],
    steps: int,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,
    show_progress: bool = True,
):
    """Train a pytree, using a loss with params, static and key as arguments.

    This can be used e.g. to fit a distribution using a variational objective, such as
    the evidence lower bound.

    Args:
        key: Jax random key.
        tree: PyTree, from which trainable parameters are found using
            ``equinox.is_inexact_array``.
        loss_fn: The loss function to optimize.
        steps: The number of optimization steps.
        learning_rate: The adam learning rate. Ignored if optimizer is provided.
        optimizer: Optax optimizer. Defaults to None.
        show_progress: Whether to show progress bar. Defaults to True.

    Returns:
        A tuple containing the trained pytree and the losses.
    """
    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    params, static = eqx.partition(
        tree,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    opt_state = optimizer.init(params)

    losses = []

    keys = tqdm(jr.split(key, steps), disable=not show_progress)

    for key in keys:
        params, opt_state, loss = step(
            params,
            static,
            key=key,
            optimizer=optimizer,
            opt_state=opt_state,
            loss_fn=loss_fn,
        )
        losses.append(loss.item())
        keys.set_postfix({"loss": loss.item()})
    return eqx.combine(params, static), losses


def fit_to_data(
    key: PRNGKeyArray,
    dist: PyTree,  # Custom losses may support broader types than AbstractDistribution
    x: ArrayLike,
    weights: ArrayLike | None = None,
    *,
    condition: ArrayLike | None = None,
    loss_fn: Callable | None = None,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    val_prop: float = 0.1,
    return_best: bool = True,
    show_progress: bool = True,
):
    r"""Train a PyTree (e.g. a distribution) to samples from the target.

    The model can be unconditional :math:`p(x)` or conditional
    :math:`p(x|\text{condition})`. Note that the last batch in each epoch is dropped
    if truncated (to avoid recompilation). This function can also be used to fit
    non-distribution pytrees as long as a compatible loss function is provided.

    Args:
        key: Jax random seed.
        dist: The pytree to train (usually a distribution).
        x: Samples from target distribution.
        weights: Weights of the target distribution samples. Defaults to None.
        learning_rate: The learning rate for adam optimizer. Ignored if optimizer is
            provided.
        optimizer: Optax optimizer. Defaults to None.
        condition: Conditioning variables. Defaults to None.
        loss_fn: Loss function. Defaults to MaximumLikelihoodLoss.
        max_epochs: Maximum number of epochs. Defaults to 100.
        max_patience: Number of consecutive epochs with no validation loss improvement
            after which training is terminated. Defaults to 5.
        batch_size: Batch size. Defaults to 100.
        val_prop: Proportion of data to use in validation set. Defaults to 0.1.
        return_best: Whether the result should use the parameters where the minimum loss
            was reached (when True), or the parameters after the last update (when
            False). Defaults to True.
        show_progress: Whether to show progress bar. Defaults to True.

    Returns:
        A tuple containing the trained distribution and the losses.
    """
    data = (x,) if condition is None else (x, condition)
    if weights is None:
        data = tuple(jnp.asarray(a) for a in data)
    else:
        data = tuple(jnp.asarray(a) for a in (jnp.c_[x, weights],))

    if loss_fn is None:
        if weights is None:
            loss_fn = MaximumLikelihoodLoss()
        else:
            loss_fn = WeightedMaximumLikelihoodLoss()

    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    params, static = eqx.partition(
        dist,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    best_params = params
    opt_state = optimizer.init(params)

    # train val split
    key, subkey = jr.split(key)
    train_data, val_data = train_val_split(subkey, data, val_prop=val_prop)
    losses = {"train": [], "val": []}

    loop = tqdm(range(max_epochs), disable=not show_progress)

    for _ in loop:
        # Shuffle data
        key, *subkeys = jr.split(key, 3)
        train_data = [jr.permutation(subkeys[0], a) for a in train_data]
        val_data = [jr.permutation(subkeys[1], a) for a in val_data]

        # Train epoch
        batch_losses = []
        for batch in zip(*get_batches(train_data, batch_size), strict=True):
            key, subkey = jr.split(key)
            if weights is None:
                params, opt_state, loss_i = step(
                    params,
                    static,
                    *batch,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    loss_fn=loss_fn,
                    key=subkey,
                )
            else:
                params, opt_state, loss_i = step(
                    params,
                    static,
                    batch[0][:,:-1], batch[0][:,-1],
                    optimizer=optimizer,
                    opt_state=opt_state,
                    loss_fn=loss_fn,
                )
            batch_losses.append(loss_i)
        losses["train"].append((sum(batch_losses) / len(batch_losses)).item())

        # Val epoch
        batch_losses = []
        for batch in zip(*get_batches(val_data, batch_size), strict=True):
            key, subkey = jr.split(key)
            if weights is None:
                loss_i = loss_fn(params, static, *batch, key=subkey)
            else:
                loss_i = loss_fn(params, static, batch[0][:,:-1], batch[0][:,-1], key=subkey)
            batch_losses.append(loss_i)
        losses["val"].append((sum(batch_losses) / len(batch_losses)).item())

        loop.set_postfix({k: v[-1] for k, v in losses.items()})
        if losses["val"][-1] == min(losses["val"]):
            best_params = params

        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

    params = best_params if return_best else params
    dist = eqx.combine(params, static)
    return dist, losses
