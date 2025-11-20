### utility functions for cosmic shear corner plots and MCMC chain processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from pathlib import Path
from getdist import MCSamples

def load_and_process_data(filepath, new_priors=None, displace_mean_frompriors=None, pop_params=None, rename_priors=None, rename_dict=None):
    """
    Load and process MCMC chain data
    
    Parameters
    ----------
    filepath : str
        Path to the MCMC chain data file.
    new_priors : dict, optional
        Dictionary of new prior distributions for reweighting. Keys are parameter names, values are tuples (mu, sigma).
    displace_mean_frompriors : dict, optional
        Dictionary of parameters to displace mean values to match new priors. Keys are parameter names, values are tuples (mu, sigma).
    pop_params : list, optional
        List of parameter names to remove from the data.
    rename_priors : dict, optional
        Dictionary for renaming prior parameters. Keys are old names, values are new names.
    rename_dict : dict, optional
        Dictionary for renaming parameters. Keys are new names, values are old names.
    flip_pz_shift : bool, optional
        Whether to flip the sign of photo-z shift parameters.
    """
    with open(filepath, "r") as f:
        header = f.readline().lstrip("#").strip().split()

    df = pd.read_csv(
        filepath, 
        sep='\s+', 
        comment="#", 
        names=header
    )
    
    if rename_priors is not None:
        for k,v in rename_priors.items():
            df[v] = df[k]

    if pop_params is not None:
        if isinstance(pop_params, str):
            pop_params = [pop_params]
        for param in pop_params:
            assert param in df.columns, f'{param} not in columns'
            df = df.drop(columns=[param])

    if new_priors is not None:
        # making the importance weights
        assert "weight" in df.columns and all(name in df.columns for name in new_priors), "Weights column missing and some prior parameters not in data"
        weights = np.ones(len(df))
        for name in new_priors:
            assert name in df.columns, f'{name} not in columns'
            #old_prior_vals = old_priors[name].pdf(df[name])
            #new_prior_vals = new_priors[name].pdf(df[name])
            mu_i, sigma_i = new_priors[name]
            weights *= np.exp(-(df[name] - mu_i)**2 / (2 * sigma_i**2))
    else:
        weights = np.ones(len(df))
    
    if displace_mean_frompriors is not None:
        for name in displace_mean_frompriors:
            assert name in df.columns, f'{name} not in columns'
            mu_i, sigma_i = displace_mean_frompriors[name]
            current_mean = df[name].mean()
            print(current_mean, mu_i)
            df[name] += (mu_i - current_mean)

    return df, header, np.asarray(weights)

def compute_effective_weights(df, prior_rescale, use_log_weight=False):
    # unless importance re-sampling, prior_rescale = np.ones(len(df))
    if use_log_weight:
        eff = np.exp(df["log_weight"].values) * prior_rescale
    else:
        eff = df["weight"].values * prior_rescale
        eff[eff == 0] = 1e-200  # avoid zero weights
    return eff

def make_mc_sample_cc(df, params, prior_rescale, burn=0., use_log_weight=False):
    eff = compute_effective_weights(df, prior_rescale, use_log_weight)
    burn_count = int(np.floor(burn * len(df))) if burn>0 else 0
    # intersect params with df columns
    params = [p for p in params if p in df.columns]
    # prior rescale can be very little
    prior = np.zeros_like(prior_rescale)
    prior[prior_rescale > 0] = np.log(prior_rescale[prior_rescale > 0])
    df['post-reweighted'] = df['post'] + prior
    df['effw'] = eff
    return df.iloc[burn_count:], eff[burn_count:]

def make_mc_sample(df, params, param_labels, prior_rescale, label, burn=0., use_log_weight=False):
    eff = compute_effective_weights(df, prior_rescale, use_log_weight)
    burn_count = int(np.floor(burn * len(df))) if burn>0 else 0
    # intersect params with df columns
    params = [p for p in params if p in df.columns]
    samples = MCSamples(samples=df[params].values, names=params, labels=param_labels, weights=eff, label=label)
    samples.removeBurn(burn)

    prior = np.zeros_like(prior_rescale)
    prior[prior_rescale > 0] = np.log(prior_rescale[prior_rescale > 0])
    df['post-reweighted'] = df['post'] + prior
    df['effw'] = eff
    return samples, df.iloc[burn_count:].reset_index(drop=True), samples.weights

def summarize_chain(samples: MCSamples, out_file: Path):
    with open(out_file, 'a') as f:
        header = f"Summary for {samples.label}\n"
        f.write(header)
        print(header.strip())
        
        for param in samples.paramNames.list():
            line =  samples.getInlineLatex(param)
            f.write(f'{param}: {line}\n')
            print(line)
        f.write("\n")
        
def confidence_interval(x, cdf, level=0.68):
    """Return the lower and upper bounds of a central confidence interval."""
    lower_prob = (1 - level) / 2
    upper_prob = 1 - lower_prob
    lower = np.interp(x=lower_prob, xp=cdf, fp=x)
    upper = np.interp(x=upper_prob, xp=cdf, fp=x)
    return lower, upper

def analyze_param(df_post_reweight, samples: MCSamples, param: str):
    """Analyze a single parameter from the posterior reweighted dataframe."""
    max_idx = np.argmax(df_post_reweight["post-reweighted"].values)
    MAP_val = df_post_reweight.iloc[max_idx][param]

    param_density = samples.get1DDensity(param)
    x = param_density.x
    P = param_density.P

    norm = np.trapezoid(P, x)
    pdf = P / norm

    posterior_mode = x[np.argmax(pdf)]
    posterior_mean = np.trapezoid(pdf * x, x)

    dx = np.mean(np.diff(x))
    cdf = np.cumsum(pdf * dx)
    cdf /= cdf[-1]

    ci_68 = confidence_interval(x, cdf, 0.68)
    ci_95 = confidence_interval(x, cdf, 0.95)

    return MAP_val, posterior_mode, posterior_mean, ci_68, ci_95

def analyze_param_old(df_post_reweight, samples: MCSamples, param: str):
    """Analyze a single parameter from the posterior reweighted dataframe."""
    max_idx = np.argmax(df_post_reweight["post-reweighted"].values)
    MAP_val = df_post_reweight.iloc[max_idx][param]

    param_density = samples.get1DDensity(param)
    x = param_density.x
    P = param_density.P
    prob = P / np.trapezoid(P, x)
    
    mode_idx = np.argmax(prob)
    posterior_mode = x[mode_idx]
    posterior_mean = np.trapezoid(prob * x, x)
    
    dx = np.mean(np.diff(x))
    print(np.sum(prob * dx))
    param_cdf = np.cumsum(prob * dx) / np.sum(prob * dx)

    ci_68 = confidence_interval(x, param_cdf, 0.68)
    ci_95 = confidence_interval(x, param_cdf, 0.95)

    return MAP_val, posterior_mode, posterior_mean, ci_68, ci_95

def summarize_samples(dfs_list, samples_list, params, filename, colors, linestyles):
    plt.figure(figsize=(12,4))
    plt.axhline(0, color='black', linestyle='--')
    with open(filename, "w") as f:
        for ins, (df, samples) in enumerate(zip(dfs_list, samples_list)):
            header = f"\n=== {samples.label} ===\n"
            print(header, end="")
            f.write(header)
            
            for p in params:
                if p in df.columns:
                    max_idx = np.argmax(df["post-reweighted"].values)
                    MAP_val = df[p].values[max_idx]
                    
                    param_density = samples.get1DDensity(p)
                    x = param_density.x
                    P = param_density.P
                    prob = P / np.trapezoid(P, x)
                    
                    mode_idx = np.argmax(prob)
                    posterior_mode = param_density.x[mode_idx]
                    posterior_mean = np.trapezoid(prob * param_density.x, param_density.x)
                    
                    dx = np.mean(np.diff(x))
                    param_cdf = np.cumsum(prob * dx) / np.sum(prob * dx)

                    ci_68 = confidence_interval(param_density.x, param_cdf, 0.68)
                    ci_95 = confidence_interval(param_density.x, param_cdf, 0.95)

                    if p == "cosmological_parameters--omega_m":
                        plt.plot(x, P, color=colors[ins], label=samples.label + rf": {posterior_mode:.3f}$^{{+{(ci_68[1] - posterior_mode):.3f}}}_{{-{(posterior_mode - ci_68[0]):.3f}}}$", linestyle=linestyles[ins])
                        plt.axvline(posterior_mode, color=colors[ins], linestyle=linestyles[ins])
                        plt.axvline(ci_68[0], color=colors[ins], linestyle=linestyles[ins], alpha=0.5)
                        plt.axvline(ci_68[1], color=colors[ins], linestyle=linestyles[ins], alpha=0.5)
                        plt.legend(loc="upper right")
                        plt.title(samples.label)
                        plt.grid()
                        plt.xlabel(p)
                        plt.ylabel("Density")
                        plt.xlim(0.1, 0.4)
                        
                    line = (
                        f"{p:35s} | Mode: {posterior_mode:.4f}, "
                        f"MAP: {MAP_val:.4f}, "
                        f"Mean: {posterior_mean:.4f} "
                        f"68%-: {ci_68[0]:.4f}, 68%+: {ci_68[1]:.4f}, "
                        f"95%-: {ci_95[0]:.4f}, 95%+: {ci_95[1]:.4f}, LaTeX: "
                        f"{samples.getLatex(p)}"
                    )
                    print(line)
                    f.write(line + "\n")

def compute_sigma_analytic(val1, sig1, val2, sig2):
    """Gaussian analytical sigma"""
    return (val1 - val2) / np.sqrt(sig1**2 + sig2**2)