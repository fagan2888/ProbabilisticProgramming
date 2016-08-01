import pymc3 as pm


def get_trace(X, y, n_samples=1000, random_seed=0):
    """
    A simple Bayesian linear regression model with normal priors.

    Paramters
    ---------
    X: A numpy array
    y: A numpy array
    n_samples: The number of samples to draw in pymc3.sample().
               Defaults to 1000.
    random_seed: An int. Used in pymc3.sample().
                 Defaults to 0.

    Returns
    -------
    A pymc3.MultiTrace object with access to sampling values.
    """

    with pm.Model() as linear_model:
        alpha = pm.Normal('alpha', mu=0.0, sd=1.0)
        beta = pm.Normal('beta', mu=10.0, sd=1.0)
        sigma = pm.Uniform('sigma', lower=0, upper=100)
        mu = alpha + beta * X
        y = pm.Normal('y', mu=mu, sd=sigma, observed=y)
        start = pm.find_MAP()
        step = pm.NUTS(scaling=start)
        trace = pm.sample(n_samples, start=start, step=step, model=linear_model, random_seed=random_seed,
                          progressbar=True)

    return trace


def poisson_posterior(X, idx, n_samples=2000, random_seed=0):
    """
    A hierarchical Poisson model.

    Paramters
    ---------
    X: A numpy array
    y: A numpy array
    n_samples: The number of samples to draw in pymc3.sample().
               Defaults to 2000.
    random_seed: An int. Used in pymc3.sample().
                 Defaults to 0.

    Returns
    -------
    A pymc3.MultiTrace object with access to sampling values.
    """

    with pm.Model() as hierarchical_model:
        hyper_alpha_mu = pm.Gamma('hyper_alpha_mu', alpha=1.0, beta=1.0)
        hyper_beta_mu = pm.Gamma('hyper_beta_mu', alpha=1.0, beta=1.0)
        mu = pm.Gamma('mu', alpha=hyper_alpha_mu, beta=hyper_beta_mu, shape=len(set(idx)))

        y_exp = pm.Poisson('y_exp', mu[idx], observed=X)
        y_pred = pm.Poisson('y_pred', mu[idx], shape=len(idx))

        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(n_samples, step, start=start, random_seed=random_seed, progressbar=True)

    return trace


def linear_posterior(X, y, n_samples=1000, random_seed=0):
    """
    A general linear model.

    Paramters
    ---------
    X: A numpy array
    y: A numpy array
    n_samples: The number of samples to draw in pymc3.sample().
               Defaults to 1000.
    random_seed: An int. Used in pymc3.sample().
                 Defaults to 0.

    Returns
    -------
    A pymc3.MultiTrace object with access to sampling values.
    """

    df = {'x': X, 'y': y}

    with pm.Model() as model_glm:
        pm.glm.glm('y ~ x', df, family=pm.glm.families.StudentT())
        start = pm.find_MAP()
        step = pm.NUTS(scaling=start)
        trace = pm.sample(n_samples, start=start, step=step, model=model_glm, random_seed=random_seed, progressbar=True)

    return trace
