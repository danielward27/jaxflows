import equinox as eqx
import jax.numpy as jnp
from jax import random
from paramax.wrappers import unwrap
import optax

from flowjax.bijections import Affine
from flowjax.distributions import Normal, Transformed
from flowjax.train.loops import fit_to_data


def test_data_fit():
    dim = 3
    mean, std = jnp.ones(dim), jnp.ones(dim)
    base_dist = Normal(mean, std)
    flow = Transformed(base_dist, Affine(jnp.ones(dim), jnp.ones(dim)))

    # All params should change by default
    before = eqx.filter(flow, eqx.is_inexact_array)
    x = random.normal(random.key(0), (100, dim))
    flow, losses = fit_to_data(
        random.key(0),
        flow,
        x,
        max_epochs=1,
        batch_size=50,
    )
    after = eqx.filter(flow, eqx.is_inexact_array)

    flow2, losses2, opt_state = fit_to_data(
        random.key(0),
        flow,
        x,
        max_epochs=1,
        batch_size=50,
        return_opt_state=True,
    )

    assert jnp.all(before.base_dist.bijection.loc != after.base_dist.bijection.loc)
    assert jnp.all(before.bijection.loc != after.bijection.loc)
    assert isinstance(losses["train"][0], float)
    assert isinstance(losses["val"][0], float)


def test_data_fit_opt_state():
    dim = 3
    mean, std = jnp.ones(dim), jnp.ones(dim)
    base_dist = Normal(mean, std)
    flow = Transformed(base_dist, Affine(jnp.ones(dim), jnp.ones(dim)))

    # All params should change by default
    before = eqx.filter(flow, eqx.is_inexact_array)
    values = random.normal(random.key(0), (100, dim))
    log_probs = random.normal(random.key(1), (100,))

    def loss_fn(params, static, values, log_probs, key=None):
        flow = unwrap(eqx.combine(params, static, is_leaf=eqx.is_inexact_array))
        return (log_probs - flow.log_prob(params, values)).mean()

    flow, losses, opt_state = fit_to_data(
        random.key(0),
        flow,
        values,
        log_probs,
        max_epochs=1,
        batch_size=50,
        return_opt_state=True,
    )
    after = eqx.filter(flow, eqx.is_inexact_array)

    assert jnp.all(before.base_dist.bijection.loc != after.base_dist.bijection.loc)
    assert jnp.all(before.bijection.loc != after.bijection.loc)
    assert isinstance(losses["train"][0], float)
    assert isinstance(losses["val"][0], float)

    # Continue training on new  data
    values = random.normal(random.key(2), (100, dim))
    log_probs = random.normal(random.key(3), (100,))

    flow, losses, opt_state = fit_to_data(
        random.key(4),
        flow,
        values,
        log_probs,
        max_epochs=1,
        batch_size=50,
        return_opt_state=True,
        opt_state=opt_state,
    )
    after = eqx.filter(flow, eqx.is_inexact_array)

    assert jnp.all(before.base_dist.bijection.loc != after.base_dist.bijection.loc)
    assert jnp.all(before.bijection.loc != after.bijection.loc)
    assert isinstance(losses["train"][0], float)
    assert isinstance(losses["val"][0], float)
