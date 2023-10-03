"""
File contains the SIR numpyro primitive.
"""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as npdist
from jax.experimental.ode import odeint


def SIR(beta, gamma, time, observed_data, z_init, n_states):
    """
    Susceptible-Infectious (SIS) model.

    Args:
        beta: Infection rate (transmission rate).
        gamma: Recovery rate.

    Returns:
        S, I, R : Arrays representing the number of susceptible, infected, and recovered individuals
                 at each time step.
    """
    def dz_dt(z, t):
        S = z[0]
        I = z[1]
        # R = z[2]

        N = 763  # hardcode N!!! -> change this in the future
        dS_dt = -beta * I * S / N
        dI_dt = beta * I * S / N - gamma * I
        dR_dt = gamma * I

        return jnp.stack([dS_dt, dI_dt, dR_dt])

    if z_init is None:
        z_init = numpyro.sample("z_init", npdist.LogNormal(jnp.log(10), 1).expand([n_states]))
    if beta is None:
        beta = numpyro.sample("beta", npdist.TruncatedNormal(loc=2, scale=1, low=0.))
    if gamma is None:
        gamma = numpyro.sample("gamma", npdist.TruncatedNormal(loc=0.4, scale=0.5, low=0.))

    # integrate dz/dt, the result will have shape num_days x 2
    z = numpyro.deterministic("z", odeint(dz_dt, z_init, time, rtol=1e-6, atol=1e-5, mxstep=1000))

    c = numpyro.deterministic("c", jnp.array([beta, gamma]))

    # Likelihood
    #  obs = numpyro.sample("observed", dist.Poisson(z[:, 1]), obs=observed_data)
    phi_inv = numpyro.sample("phi_inv", npdist.Exponential(5))
    phi = numpyro.deterministic("phi", 1./phi_inv)
    # TODO: Check z[:, 1] thing
    obs = numpyro.sample("observed", npdist.NegativeBinomial2(z[:, 1], phi), obs=observed_data)
