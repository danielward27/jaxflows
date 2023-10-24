"""Distributions, including the abstract and concrete classes."""
import inspect
from abc import abstractmethod
from math import prod
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from equinox import AbstractVar
from jax import Array
from jax.experimental import checkify
from jax.lax import stop_gradient
from jax.scipy import stats as jstats
from jax.typing import ArrayLike

from flowjax.bijections import AbstractBijection, Affine, Chain
from flowjax.utils import _get_ufunc_signature, arraylike_to_array, merge_cond_shapes


class AbstractDistribution(eqx.Module):
    """Abstract distribution class.

    Distributions are registered as jax PyTrees (as they are equinox modules), and as
    such they are compatible with normal jax operations.

    Attributes:
        shape (AbstractVar[tuple[int, ...]]): Denotes the shape of a single sample from
            the distribution.
        cond_shape (AbstractVar[tuple[int, ...] | None]): The shape of an instance of
            the conditioning variable. This should be None for unconditional
            distributions.
    """

    shape: AbstractVar[tuple[int, ...]]
    cond_shape: AbstractVar[tuple[int, ...] | None]

    @abstractmethod
    def _log_prob(self, x: Array, condition: Array | None = None) -> Array:
        """Evaluate the log probability of point x.

        This method should be be valid for inputs with shapes matching
        ``distribution.shape`` and ``distribution.cond_shape`` for conditional
        distributions (i.e. the method defined for unbatched inputs).
        """

    @abstractmethod
    def _sample(self, key: Array, condition: Array | None = None) -> Array:
        """Sample a point from the distribution.

        This method should return a single sample with shape matching
        ``distribution.shape``.
        """

    @abstractmethod
    def _sample_and_log_prob(
        self,
        key: Array,
        condition: Array | None = None,
    ) -> tuple[Array, Array]:
        """Sample a point from the distribution, and return its log probability."""

    def log_prob(self, x: ArrayLike, condition: ArrayLike | None = None) -> Array:
        """Evaluate the log probability.

        Uses numpy-like broadcasting if additional leading dimensions are passed.

        Args:
            x (ArrayLike): Points at which to evaluate density.
            condition (ArrayLike | None): Conditioning variables. Defaults to None.

        Returns:
            Array: Jax array of log probabilities.
        """
        x = arraylike_to_array(x, "x")
        condition = self._argcheck_and_cast_condition(condition)
        lps = self._vectorize(self._log_prob)(x, condition)
        return jnp.where(jnp.isnan(lps), -jnp.inf, lps)

    def sample(
        self,
        key: Array,
        sample_shape: tuple[int, ...] = (),
        condition: ArrayLike | None = None,
    ) -> Array:
        """Sample from the distribution.

        For unconditional distributions, the output will be of shape
        ``sample_shape + dist.shape``. For conditional distributions, a batch dimension
        in the condition is supported, and the output shape will be
        ``sample_shape + condition_batch_shape + dist.shape``.
        See the example for more information.

        Args:
            key (Array): Jax random key.
            condition (ArrayLike | None): Conditioning variables. Defaults to None.
            sample_shape (tuple[int, ...]): Sample shape. Defaults to ().

        Example:
            The below example shows the behaviour of sampling, for an unconditional
            and a conditional distribution.

            .. testsetup::

                from flowjax.distributions import StandardNormal
                import jax.random as jr
                import jax.numpy as jnp
                from flowjax.flows import CouplingFlow
                from flowjax.bijections import Affine
                # For a unconditional distribution:
                key = jr.PRNGKey(0)
                dist = StandardNormal((2,))
                # For a conditional distribution
                cond_dist = CouplingFlow(
                    key, StandardNormal((2,)), cond_dim=3, transformer=Affine()
                    )

            For an unconditional distribution:

            .. doctest::

                >>> dist.shape
                (2,)
                >>> samples = dist.sample(key, (10, ))
                >>> samples.shape
                (10, 2)

            For a conditional distribution:

            .. doctest::

                >>> cond_dist.shape
                (2,)
                >>> cond_dist.cond_shape
                (3,)
                >>> # Sample 10 times for a particular condition
                >>> samples = cond_dist.sample(key, (10,), condition=jnp.ones(3))
                >>> samples.shape
                (10, 2)
                >>> # Sampling, batching over a condition
                >>> samples = cond_dist.sample(key, condition=jnp.ones((5, 3)))
                >>> samples.shape
                (5, 2)
                >>> # Sample 10 times for each of 5 conditioning variables
                >>> samples = cond_dist.sample(key, (10,), condition=jnp.ones((5, 3)))
                >>> samples.shape
                (10, 5, 2)


        """
        condition = self._argcheck_and_cast_condition(condition)
        keys = self._get_sample_keys(key, sample_shape, condition)
        return self._vectorize(self._sample)(keys, condition)

    def sample_and_log_prob(
        self,
        key: Array,
        sample_shape: tuple[int, ...] = (),
        condition: ArrayLike | None = None,
    ):
        """Sample the distribution and return the samples with their log probabilities.

        For transformed distributions (especially flows), this will generally be more
        efficient than calling the methods seperately. Refer to the
        :py:meth:`~flowjax.distributions.AbstractDistribution.sample` documentation for
        more information.

        Args:
            key (Array): Jax random key.
            condition (ArrayLike | None): Conditioning variables. Defaults to None.
            sample_shape (tuple[int, ...]): Sample shape. Defaults to ().
        """
        condition = self._argcheck_and_cast_condition(condition)
        keys = self._get_sample_keys(key, sample_shape, condition)
        return self._vectorize(self._sample_and_log_prob)(keys, condition)

    @property
    def ndim(self):
        """Number of dimensions in the distribution (the length of the shape)."""
        return len(self.shape)

    @property
    def cond_ndim(self):
        """Number of dimensions of the conditioning variable (length of cond_shape)."""
        return None if self.cond_shape is None else len(self.cond_shape)

    def _argcheck_and_cast_condition(self, condition: ArrayLike | None):
        if self.cond_shape is None:
            if condition is not None:
                raise TypeError(
                    f"Expected condition to be None; got {type(condition)}.",
                )
            return None
        return arraylike_to_array(condition, err_name="condition")

    def _vectorize(self, method: callable) -> callable:
        """Returns a vectorized version of the distribution method."""
        # Get shapes without broadcasting - note the (2, ) corresponds to key arrays.
        maybe_cond = [] if self.cond_shape is None else [self.cond_shape]
        in_shapes = {
            "_sample_and_log_prob": [(2,)] + maybe_cond,
            "_sample": [(2,)] + maybe_cond,
            "_log_prob": [self.shape] + maybe_cond,
        }
        out_shapes = {
            "_sample_and_log_prob": [self.shape, ()],
            "_sample": [self.shape],
            "_log_prob": [()],
        }
        in_shapes, out_shapes = in_shapes[method.__name__], out_shapes[method.__name__]

        def _check_shapes(method):
            # Wraps unvectorised method with shape checking
            def _wrapper(*args):
                argnames = list(inspect.signature(method).parameters)
                assert len(args) == len(argnames)
                for in_shape, arg, name in zip(in_shapes, args, argnames, strict=False):
                    if arg.shape != in_shape:
                        raise ValueError(
                            f"Expected trailing dimensions matching {in_shape} for "
                            f"{name}; got {arg.shape}.",
                        )
                return method(*args)

            return _wrapper

        signature = _get_ufunc_signature(in_shapes, out_shapes)
        ex = frozenset([1]) if self.cond_shape is None else frozenset()
        return jnp.vectorize(_check_shapes(method), signature=signature, excluded=ex)

    def _get_sample_keys(self, key, sample_shape, condition):
        if self.cond_shape is not None:
            leading_cond_shape = condition.shape[: -self.cond_ndim or None]
        else:
            leading_cond_shape = ()
        key_shape = sample_shape + leading_cond_shape
        key_size = max(1, prod(key_shape))  # Still need 1 key for scalar sample
        return jnp.reshape(jr.split(key, key_size), (*key_shape, 2))


class AbstractStandardDistribution(AbstractDistribution):
    """Abstract distribution type, representing a distribution that is not transformed.

    The class implements a default ``_sample_and_log_prob`` method by calling
    ``_sample`` and ``_log_prob`` methods seperately. Concrete subclasses can be
    implemented as follows:

        (1) Inherit from :class:`AbstractStandardDistribution`.
        (2) Define the abstract attributes ``shape`` and ``cond_shape``.
            ``cond_shape`` should be ``None`` for unconditional distributions.
        (3) Define the abstract methods :meth:`_sample` and :meth:`_log_prob`.

    See the source code for :class:`StandardNormal` for a simple example implementation.
    """

    def _sample_and_log_prob(self, key, condition=None):
        x = self._sample(key, condition)
        return x, self._log_prob(x, condition)


class AbstractTransformed(AbstractDistribution):
    """Abstract class respresenting transformed distributions.

    Concete implementations can be implemnted as follows:

        (1) Inherit from :class:`AbstractTransformed`.
        (2) Define the abstract attributes `base_dist`, `bijection`.

    bijection attributes. We take the forward bijection for use in sampling, and the
    inverse for use in density evaluation. See also :class:`Transformed`

    """

    base_dist: AbstractVar[AbstractDistribution]
    bijection: AbstractVar[AbstractBijection]

    def _log_prob(self, x, condition=None):
        z, log_abs_det = self.bijection.inverse_and_log_det(x, condition)
        p_z = self.base_dist._log_prob(z, condition)
        return p_z + log_abs_det

    def _sample(self, key, condition=None):
        base_sample = self.base_dist._sample(key, condition)
        return self.bijection.transform(base_sample, condition)

    def _sample_and_log_prob(self, key: Array, condition=None):
        # We avoid computing the inverse transformation.
        base_sample, log_prob_base = self.base_dist._sample_and_log_prob(key, condition)
        sample, forward_log_dets = self.bijection.transform_and_log_det(
            base_sample,
            condition,
        )
        return sample, log_prob_base - forward_log_dets

    def __check_init__(self):  # TODO test errors and test conditional base distribution
        """Checks cond_shape is compatible in both bijection and distribution."""
        if (
            self.base_dist.cond_shape is not None
            and self.bijection.cond_shape is not None
            and self.base_dist.cond_shape != self.bijection.cond_shape
        ):
            raise ValueError(
                "The base distribution and bijection are both conditional "
                "but have mismatched cond_shape attributes. Base distribution has"
                f"{self.base_dist.cond_shape}, and the bijection has"
                f"{self.bijection.cond_shape}.",
            )

    def merge_transforms(self):
        """Unnests nested transformed distributions.

        Returns an equivilent distribution, but ravelling nested
        :class:`AbstractTransformed` distributions such that the returned distribution
        has a base distribution that is not an :class:`AbstractTransformed` instance.
        """
        if not isinstance(self.base_dist, AbstractTransformed):
            return self
        base_dist = self.base_dist
        bijections = [self.bijection]
        while isinstance(base_dist, AbstractTransformed):
            bijections.append(base_dist.bijection)
            base_dist = base_dist.base_dist
        bijection = Chain(list(reversed(bijections))).merge_chains()
        return Transformed(base_dist, bijection)

    @property
    def shape(self):
        return self.base_dist.shape

    @property
    def cond_shape(self):
        return merge_cond_shapes((self.bijection.cond_shape, self.base_dist.cond_shape))


class Transformed(AbstractTransformed):
    """Form a distribution like object using a base distribution and a bijection.

    We take the forward bijection for use in sampling, and the inverse
    bijection for use in density evaluation.

    .. warning::
            It is the currently the users responsibility to ensure the bijection is
            valid across the entire support of the distribution. Failure to do so may
            lead to to unexpected results.
    """

    base_dist: AbstractDistribution
    bijection: AbstractBijection

    def __init__(
        self,
        base_dist: AbstractDistribution,
        bijection: AbstractBijection,
    ):
        """Initialize the transformed distribution.

        Args:
            base_dist (AbstractDistribution): Base distribution.
            bijection (AbstractBijection): Bijection to transform distribution.

        Example:
            .. doctest::

                >>> from flowjax.distributions import StandardNormal, Transformed
                >>> from flowjax.bijections import Affine
                >>> normal = StandardNormal()
                >>> bijection = Affine(1)
                >>> transformed = Transformed(normal, bijection)
        """
        self.base_dist = base_dist
        self.bijection = bijection


class StandardNormal(AbstractStandardDistribution):
    """Standard normal distribution.

    Note unlike :class:`Normal`, this has no trainable parameters.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(self, shape: tuple[int, ...] = ()):
        """Initialize the standard Normal distribution.

        Args:
            shape (tuple[int, ...]): The shape of the distribution. Defaults to ().
        """
        self.shape = shape

    def _log_prob(self, x, condition=None):
        return jstats.norm.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.normal(key, self.shape)


class Normal(AbstractTransformed):
    """An independent Normal distribution with mean and std for each dimension."""

    base_dist: StandardNormal
    bijection: Affine
    cond_shape: ClassVar[None] = None

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """Initialize the normal distribution.

        ``loc`` and ``scale`` should broadcast to the desired shape of the distribution.

        Args:
            loc (ArrayLike): Means. Defaults to 0.
            scale (ArrayLike): Standard deviations. Defaults to 1.
        """
        self.base_dist = StandardNormal(
            jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale)),
        )
        self.bijection = Affine(loc=loc, scale=scale)

    @property
    def loc(self):
        """Location of the distribution."""
        return self.bijection.loc

    @property
    def scale(self):
        """Scale of the distribution."""
        return self.bijection.scale


class _StandardUniform(AbstractStandardDistribution):
    r"""Implements a standard Uniform distribution."""
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(self, shape: tuple[int, ...] = ()):
        self.shape = shape

    def _log_prob(self, x, condition=None):
        return jstats.uniform.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.uniform(key, shape=self.shape)


class Uniform(AbstractTransformed):
    """Independent uniform distribution."""

    base_dist: _StandardUniform
    bijection: Affine

    def __init__(self, minval: ArrayLike, maxval: ArrayLike):
        """Initialize the uniform distribution.

        ``minval`` and ``maxval`` should broadcast to the desired distribution shape.

        Args:
            minval (ArrayLike): Minimum values.
            maxval (ArrayLike): Maximum values.
        """
        minval, maxval = arraylike_to_array(minval), arraylike_to_array(maxval)
        checkify.check(
            jnp.all(maxval >= minval),
            "Minimums must be less than the maximums.",
        )
        self.base_dist = _StandardUniform(
            jnp.broadcast_shapes(minval.shape, maxval.shape),
        )
        self.bijection = Affine(loc=minval, scale=maxval - minval)

    @property
    def minval(self):
        """Minimum value of the uniform distribution."""
        return self.bijection.loc

    @property
    def maxval(self):
        """Maximum value of the uniform distribution."""
        return self.bijection.loc + self.bijection.scale


class _StandardGumbel(AbstractStandardDistribution):
    """Standard gumbel distribution (https://en.wikipedia.org/wiki/Gumbel_distribution)."""

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(self, shape: tuple[int, ...] = ()):
        self.shape = shape

    def _log_prob(self, x, condition=None):
        return -(x + jnp.exp(-x)).sum()

    def _sample(self, key, condition=None):
        return jr.gumbel(key, shape=self.shape)


class Gumbel(AbstractTransformed):
    """Gumbel distribution (https://en.wikipedia.org/wiki/Gumbel_distribution)."""

    base_dist: _StandardGumbel
    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """``loc`` and ``scale`` should broadcast to the dimension of the distribution.

        Args:
            loc (ArrayLike): Location paramter.
            scale (ArrayLike): Scale parameter. Defaults to 1.0.
        """
        self.base_dist = _StandardGumbel(
            jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale)),
        )
        self.bijection = Affine(loc, scale)

    @property
    def loc(self):
        """Location of the distribution."""
        return self.bijection.loc

    @property
    def scale(self):
        """Scale of the distribution."""
        return self.bijection.scale


class _StandardCauchy(AbstractStandardDistribution):
    """Implements standard cauchy distribution (loc=0, scale=1).

    Ref: https://en.wikipedia.org/wiki/Cauchy_distribution.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(self, shape: tuple[int, ...] = ()):
        self.shape = shape

    def _log_prob(self, x, condition=None):
        return jstats.cauchy.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.cauchy(key, shape=self.shape)


class Cauchy(AbstractTransformed):
    """Cauchy distribution (https://en.wikipedia.org/wiki/Cauchy_distribution)."""

    base_dist: _StandardCauchy
    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """``loc`` and ``scale`` should broadcast to the dimension of the distribution.

        Args:
            loc (ArrayLike): Location paramter.
            scale (ArrayLike): Scale parameter. Defaults to 1.0.
        """
        self.base_dist = _StandardCauchy(
            jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale)),
        )
        self.bijection = Affine(loc, scale)

    @property
    def loc(self):
        """Location of the distribution."""
        return self.bijection.loc

    @property
    def scale(self):
        """Scale of the distribution."""
        return self.bijection.scale


class _StandardStudentT(AbstractStandardDistribution):
    """Implements student T distribution with specified degrees of freedom."""

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    log_df: Array

    def __init__(self, df: ArrayLike):
        self.shape = jnp.shape(df)
        self.log_df = jnp.log(df)  # TODO post init check of not nans?

    def _log_prob(self, x, condition=None):
        return jstats.t.logpdf(x, df=self.df).sum()

    def _sample(self, key, condition=None):
        return jr.t(key, df=self.df, shape=self.shape)

    @property
    def df(self):
        """The degrees of freedom of the distibution."""
        return jnp.exp(self.log_df)


class StudentT(AbstractTransformed):
    """Student T distribution (https://en.wikipedia.org/wiki/Student%27s_t-distribution)."""

    base_dist: _StandardStudentT
    bijection: Affine

    def __init__(self, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1):
        """``df``, ``loc`` and ``scale`` broadcast to the dimension of the distribution.

        Args:
            df (ArrayLike): The degrees of freedom.
            loc (ArrayLike): Location parameter. Defaults to 0.0.
            scale (ArrayLike): Scale parameter. Defaults to 1.0.
        """
        df, loc, scale = jnp.broadcast_arrays(df, loc, scale)
        self.base_dist = _StandardStudentT(df)
        self.bijection = Affine(loc, scale)

    @property
    def loc(self):
        """Location of the distribution."""
        return self.bijection.loc

    @property
    def scale(self):
        """Scale of the distribution."""
        return self.bijection.scale

    @property
    def df(self):
        """The degrees of freedom of the distribution."""
        return self.base_dist.df


class SpecializeCondition(AbstractDistribution):  # TODO check tested
    """Specialise a distribution to a particular conditioning variable instance.

    This makes the distribution act like an unconditional distribution, i.e. the
    distribution methods implicitly will use the condition passed on instantiation
    of the class.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(
        self,
        dist: AbstractDistribution,
        condition: ArrayLike,
        stop_gradient: bool = True,
    ):
        """Initialize the distribution.

        Args:
            dist (AbstractDistribution): Conditional distribution to specialize.
            condition (ArrayLike, optional): Instance of conditioning variable with
                shape matching ``dist.cond_shape``. Defaults to None.
            stop_gradient (bool): Whether to use ``jax.lax.stop_gradient`` to prevent
                training of the condition array. Defaults to True.
        """
        condition = arraylike_to_array(condition)
        if self.dist.cond_shape != condition.shape:
            raise ValueError(
                f"Expected condition shape {self.dist.cond_shape}, got "
                f"{condition.shape}",
            )
        self.dist = dist
        self._condition = condition
        self.shape = dist.shape
        self.stop_gradient = stop_gradient

    def _log_prob(self, x, condition=None):
        return self.dist._log_prob(x, self.condition)

    def _sample(self, key, condition=None):
        return self.dist._sample(key, self.condition)

    def _sample_and_log_prob(self, key, condition=None):
        return self.dist._sample_and_log_prob(key, self.condition)

    @property
    def condition(self):
        """The conditioning variable, possibly with stop_gradient applied."""
        return stop_gradient(self._condition) if self.stop_gradient else self._condition
