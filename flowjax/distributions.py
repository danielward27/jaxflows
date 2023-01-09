# Distribution object (for flows and base distributions)

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp

import jax.random as jr
from jax.scipy import stats as jstats

from flowjax.bijections import Affine, Bijection
from flowjax.utils import Array, _get_ufunc_signature
from math import prod
from flowjax.utils import broadcast_shapes
# To construct a distribution, we define _log_prob and _sample, which take in vector arguments.
# More friendly methods are then created from these, supporting batches of inputs.
# Note that unconditional distributions should allow, but ignore the passing of conditional variables
# (to facilitate easy composing of conditional and unconditional distributions)

# TODO Can we use numpy vectorise on the distributions????
class Distribution(eqx.Module, ABC):
    """Distribution base class.
    """

    shape: Tuple[int]
    cond_shape: Union[None, Tuple[int]]

    @abstractmethod
    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        "Evaluate the log probability of point x."
        pass

    @abstractmethod
    def _sample(self, key: jr.KeyArray, condition: Optional[Array] = None):
        "Sample a point from the distribution."
        pass


    def log_prob(self, x: Array, condition: Optional[Array] = None):
        """Evaluate the log probability. Uses numpy like broadcasting if additional
        leading dimensions are passed. in the arguments.

        Args:
            x (Array): Points at which to evaluate density.
            condition (Optional[Array], optional): Conditioning variables. Defaults to None.

        Returns:
            Array: Jax array of log probabilities.
        """
        self._argcheck(x, condition)
        if condition is not None:
            sig = _get_ufunc_signature([self.shape, self.cond_shape], [()])
            exclude = {},
        else:
            sig = _get_ufunc_signature([self.shape], [()])
            exclude = {1}
        
        return jnp.vectorize(self._log_prob, signature=sig, excluded=exclude)(x, condition)

    def sample(self, key: jr.PRNGKey, condition: Optional[Array] = None, sample_shape: Tuple[int] = ()):
        "Output shape will be sample_shape + condition_batch_shape + self.shape"
        self._argcheck(condition=condition)

        if condition is None:
            key_shape = sample_shape
            excluded = {1}
            sig = _get_ufunc_signature([(2,)], [self.shape])
        else:
            key_shape = sample_shape + condition.shape[:-len(self.condition_shape)]
            excluded = {}
            sig = _get_ufunc_signature([(2,), self.cond_shape], [self.x_shape])
        
        key_size = max(1, prod(key_shape)) # max 1 required for scalar input
        keys = jnp.reshape(jr.split(key, key_size), key_shape + (2,))

        return jnp.vectorize(self._sample, excluded=excluded, signature=sig)(keys, condition)
    
    def _argcheck(self, x=None, condition=None):
        # jnp.vectorize will catch ndim mismatches, but it doesn't check axis lengths.
        if x is not None:
            x_trailing = x.shape[-self.ndim:] if self.ndim>0 else ()
            if  x_trailing != self.shape:
                raise ValueError(
                    f"Expected trailing dimensions in input x to match the distribution shape, but got"
                    f"x shape {x.shape}, and distribution shape {self.shape}."
                )

        if condition is None and self.cond_shape is not None:
            raise ValueError(
                f"Conditioning variable was not provided."
                f"Expected conditioning variable with trailing shape {self.shape}."
            )

        if condition is not None:
            condition_trailing = condition.shape[-self.cond_ndim:] if self.cond_ndim>0 else ()
            if condition_trailing != self.cond_shape:
                raise ValueError(
                    f"Expected trailing dimensions in the condition to match the distribution.cond_shape, but got"
                    f"condition shape {condition.shape}, and distribution.cond_shape {self.cond_shape}."
                )

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def cond_ndim(self):
        return len(self.cond_shape)


class Transformed(Distribution):
    base_dist: Distribution
    bijection: Bijection
    cond_shape: Union[None, Tuple[int]]

    def __init__(
        self,
        base_dist: Distribution,
        bijection: Bijection,
    ):
        """
        Form a distribution like object using a base distribution and a
        bijection. We take the forward bijection for use in sampling, and the inverse
        bijection for use in density evaluation.

        Args:
            base_dist (Distribution): Base distribution.
            bijection (Bijection): Bijection to transform distribution.
        """
        self.base_dist = base_dist
        self.bijection = bijection
        self.shape = self.base_dist.shape
        self.cond_shape = broadcast_shapes((self.bijection.cond_shape, self.base_dist.cond_shape))

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        z, log_abs_det = self.bijection.inverse_and_log_abs_det_jacobian(x, condition)
        p_z = self.base_dist._log_prob(z, condition)
        return p_z + log_abs_det

    def _sample(self, key: jr.KeyArray, condition: Optional[Array] = None):
        z = self.base_dist._sample(key, condition)
        x = self.bijection.transform(z, condition)
        return x


class StandardNormal(Distribution):

    def __init__(self, shape: Tuple[int] = ()):
        """
        Implements a standard normal distribution, condition is ignored.

        Args:
            dim (int): Dimension of the normal distribution. 
        """  # TODO update documentation
        self.shape = shape
        self.cond_shape = None

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jstats.norm.logpdf(x).sum()

    def _sample(self, key: jr.KeyArray, condition: Optional[Array] = None):
        return jr.normal(key, self.shape)


class Normal(Transformed):
    """
    Implements an independent Normal distribution with mean and std for
    each dimension. `loc` and `scale` should be broadcastable.
    """

    def __init__(self, loc: Array, scale: Array = 1.0):
        """
        Args:
            loc (Array): Array of the means of each dimension.
            scale (Array): Array of the standard deviations of each dimension.
        """
        loc, scale = jnp.broadcast_arrays(loc, scale)
        self.shape = loc.shape
        self.cond_shape = None
        base_dist = StandardNormal(self.shape)
        bijection = Affine(loc=loc, scale=scale)
        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        return self.bijection.loc

    @property
    def scale(self):
        return self.bijection.scale


class _StandardUniform(Distribution):
    """
    Implements a standard independent Uniform distribution, ie X ~ Uniform([0, 1]^dim).
    """

    def __init__(self, shape: Tuple[int] = ()):
        self.shape = shape
        self.cond_shape = None

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jstats.uniform.logpdf(x).sum()

    def _sample(self, key: jr.KeyArray, condition: Optional[Array] = None):
        return jr.uniform(key, shape=self.shape)


from jax.experimental import checkify

class Uniform(Transformed):
    """
    Implements an independent uniform distribution
    between min and max for each dimension. `minval` and `maxval` should be broadcastable.
    """

    def __init__(self, minval: Array, maxval: Array):
        """
        Args:
            minval (Array): ith entry gives the min of the ith dimension
            maxval (Array): ith entry gives the max of the ith dimension
        """
        minval, maxval = jnp.broadcast_arrays(minval, maxval)
        self.shape = minval.shape
        self.cond_shape = None

        checkify.check(
            jnp.all(maxval >= minval),
            "Minimums must be less than the maximums."
        )
        
        base_dist = _StandardUniform(minval.shape)
        bijection = Affine(loc=minval, scale=maxval - minval)
        super().__init__(base_dist, bijection)

    @property
    def minval(self):
        return self.bijection.loc

    @property
    def maxval(self):
        return self.bijection.loc + self.bijection.scale


class _StandardGumbel(Distribution):
    """Standard gumbel distribution (https://en.wikipedia.org/wiki/Gumbel_distribution).
    """

    def __init__(self, shape: Tuple[int] = ()):
        
        self.shape = shape
        self.cond_shape = None

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return -(x + jnp.exp(-x)).sum()

    def _sample(self, key: jr.KeyArray, condition: Optional[Array] = None):
        return jr.gumbel(key, shape=self.shape)


class Gumbel(Transformed):
    """Gumbel distribution (https://en.wikipedia.org/wiki/Gumbel_distribution)"""

    def __init__(self, loc: Array, scale: Array = 1.0):
        """
        `loc` and `scale` should broadcast to the dimension of the distribution.

        Args:
            loc (Array): Location paramter. 
            scale (Array, optional): Scale parameter. Defaults to 1.0.
        """
        loc, scale = jnp.broadcast_arrays(loc, scale)
        self.shape = loc.shape
        self.cond_shape = None
        base_dist = _StandardGumbel(self.shape)
        bijection = Affine(loc, scale)


        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        return self.bijection.loc

    @property
    def scale(self):
        return self.bijection.scale


class _StandardCauchy(Distribution):
    """
    Implements standard cauchy distribution (loc=0, scale=1)
    Ref: https://en.wikipedia.org/wiki/Cauchy_distribution
    """

    def __init__(self, shape: Tuple[int] = ()):
        self.shape = shape
        self.cond_shape = None

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jstats.cauchy.logpdf(x).sum()

    def _sample(self, key: jr.KeyArray, condition: Optional[Array] = None):
        return jr.cauchy(key, shape=self.shape)


class Cauchy(Transformed):
    """
    Cauchy distribution (https://en.wikipedia.org/wiki/Cauchy_distribution).
    """

    def __init__(self, loc: Array, scale: Array = 1.0):
        """
        `loc` and `scale` should broadcast to the dimension of the distribution.

        Args:
            loc (Array): Location paramter. 
            scale (Array, optional): Scale parameter. Defaults to 1.0.
        """
        loc, scale = jnp.broadcast_arrays(loc, scale)
        self.shape = loc.shape
        self.cond_shape = None
        base_dist = _StandardCauchy(self.shape)
        bijection = Affine(loc, scale)
        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        return self.bijection.loc

    @property
    def scale(self):
        return self.bijection.scale


class _StandardStudentT(Distribution):
    """
    Implements student T distribution with specified degrees of freedom.
    """

    log_df: Array

    def __init__(self, df: Array):
        self.shape = df.shape
        self.cond_shape = None
        self.log_df = jnp.log(df)

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jstats.t.logpdf(x, df=self.df).sum()

    def _sample(self, key: jr.KeyArray, condition: Optional[Array] = None):
        return jr.t(key, df=self.df, shape=self.shape)

    @property
    def df(self):
        return jnp.exp(self.log_df)


class StudentT(Transformed):
    """Student T distribution (https://en.wikipedia.org/wiki/Student%27s_t-distribution)."""

    def __init__(self, df: Array, loc: Array = 0.0, scale: Array = 1.0):
        """
        `df`, `loc` and `scale` broadcast to the dimension of the distribution.

        Args:
            df (Array): The degrees of freedom.
            loc (Array): Location parameter. Defaults to 0.0.
            scale (Array, optional): Scale parameter. Defaults to 1.0.
        """
        df, loc, scale = jnp.broadcast_arrays(df, loc, scale)
        self.shape = df.shape
        self.cond_shape = None
        base_dist = _StandardStudentT(df)
        bijection = Affine(loc, scale)
        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        return self.bijection.loc

    @property
    def scale(self):
        return self.bijection.scale

    @property
    def df(self):
        return self.base_dist.df
