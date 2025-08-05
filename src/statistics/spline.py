import numpy as np 
import importlib
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

from scipy.optimize import minimize, differential_evolution, nnls
from pathlib import Path
from scipy import interpolate, integrate, stats


class BayesianBSpline:
    """
    Bayesian B-splines using PyMC for modeling n(z) distributions
    with Dirichlet priors on coefficients and a free amplitude parameter.
    """
    def __init__(self, zv, n_knots=20, degree=3):
        self.zv = np.asarray(zv)
        self.n_knots = n_knots
        self.degree = degree
        self.basis_matrix = None
        self.basis_integrals = None
        self.knots = None
        self.trace = None
        self.model = None
        self.nz = None
        self.nz_err = None

    def _create_spline_basis(self):
        """Create B-spline basis functions"""
        z_min, z_max = self.zv.min(), self.zv.max()
        interior_knots = np.linspace(z_min, z_max, self.n_knots)
        self.knots = np.concatenate([
            np.repeat(interior_knots[0], self.degree),
            interior_knots,
            np.repeat(interior_knots[-1], self.degree)
        ])
        n_basis = len(interior_knots) + self.degree - 1
        basis_matrix = np.zeros((len(self.zv), n_basis))
        
        for i in range(n_basis):
            coeffs = np.zeros(n_basis)
            coeffs[i] = 1.0
            spline = interpolate.BSpline(self.knots, coeffs, self.degree)
            basis_matrix[:, i] = spline(self.zv)
        
        self.basis_matrix = basis_matrix
        self.n_basis = n_basis

    def _get_initial_coeffs_and_amplitude(self, nz):
        """Get reasonable initial coefficients and amplitude using non-negative least squares"""
        coeffs_raw, residuals = nnls(self.basis_matrix, nz)
        initial_amplitude = coeffs_raw.sum()
        return coeffs_raw, initial_amplitude

    def _compute_basis_integrals(self):
        """Compute integrals of each basis function for amplitude scaling"""
        self.basis_integrals = np.array([
            np.trapezoid(self.basis_matrix[:, i], self.zv) 
            for i in range(self.n_basis)
        ])

    def fit(
            self, 
            nz, 
            nz_err, 
            n_samples=4000, 
            n_tune=1000, 
            n_chains=4, 
            target_accept=0.95, 
            prior_concentration=1.0,
            base_alpha=0.15
            ):
        """
        Fit the model using PyMC with Dirichlet prior and free amplitude parameter.

        Parameters:
        -----------
        nz : array_like
            Data values at zv points (n(z) values)
        nz_err : array_like
            Uncertainties in nz
        prior_concentration : float
            Dirichlet concentration (higher = less sparse, should be >= 1.0)
        base_alpha : float
            Base alpha for Dirichlet prior, controls sparsity of coefficients.
            Should be a small positive value (e.g., 0.1 or 0.15).
        n_samples : int
            Number of posterior samples to draw
        n_tune : int
            Number of tuning steps for the sampler
        n_chains : int
            Number of chains to run in the sampler
        target_accept : float
            Target acceptance rate for the sampler (default 0.95)
        """
        self.nz = np.asarray(nz)
        self.nz_err = np.asarray(nz_err)

        assert len(self.nz) == len(self.zv), "nz and zv must have the same length"
        assert len(self.nz_err) == len(self.zv), "nz_err and zv must have the same length"
        assert all(self.nz_err > 0), "nz_err must be positive"
        
        self._create_spline_basis()
        self._compute_basis_integrals()
        coeffs_init, initial_amplitude = self._get_initial_coeffs_and_amplitude(self.nz)
        with pm.Model() as model:
            # Set Dirichlet alpha based on NNLS results
            # Higher alpha where NNLS found signal, lower where it found zero
            normalized_coeffs = coeffs_init / coeffs_init.sum()
            dirichlet_alpha = base_alpha + prior_concentration * normalized_coeffs
            
            print(f"Dirichlet alpha range: [{dirichlet_alpha.min():.3f}, {dirichlet_alpha.max():.3f}]")
            print(f"Non-zero NNLS coefficients: {np.sum(normalized_coeffs > 1e-10)} / {self.n_basis}")
            
            coeffs = pm.Dirichlet(
                'coeffs',
                a=dirichlet_alpha,
                shape=self.n_basis
            )
            amplitude = pm.Uniform(
                'amplitude',
                lower=0.5*initial_amplitude,
                upper=1.5*initial_amplitude,
                initval=initial_amplitude
            )
            nz_pred = pm.math.dot(self.basis_matrix, coeffs) * amplitude
            likelihood = pm.Normal(
                'likelihood', 
                mu=nz_pred, 
                sigma=self.nz_err, 
                observed=self.nz
            )
            trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                chains=n_chains,
                return_inferencedata=True,
                target_accept=target_accept,
                progressbar=True,
                random_seed=123
            )

        self.trace = trace
        self.model = model
        self.coeffs_samples = trace.posterior['coeffs'].values.reshape(-1, self.n_basis)
        self.amplitude_samples = trace.posterior['amplitude'].values.reshape(-1)
        
        summary = az.summary(trace, var_names=['coeffs', 'amplitude'])
        print(f"Model fitting complete. Summary: {summary}")

        if summary['r_hat'].max() > 1.1:
            print("Warning: R-hat > 1.1 detected. Consider more tuning or samples.")

        return self

    def _create_evaluation_basis(self, z_eval):
        """Create basis matrix for evaluation points using the same knots"""
        z_eval = np.asarray(z_eval)
        basis_eval = np.zeros((len(z_eval), self.n_basis))
        for i in range(self.n_basis):
            coeffs = np.zeros(self.n_basis)
            coeffs[i] = 1.0
            spline = interpolate.BSpline(self.knots, coeffs, self.degree)
            basis_eval[:, i] = spline(z_eval)
        return basis_eval
    
    def get_spline_from_trace(self, n_eval_points=200):
        """
        Plot the data, fitted model, and uncertainty bands
        """
        if self.trace is None:
            raise ValueError("Model must be fitted before plotting. Call fit() first.")

        if z_eval is None:
            z_eval = np.linspace(self.zv.min(), self.zv.max(), n_eval_points)
        else:
            z_eval = np.asarray(z_eval)

        basis_eval = self._create_evaluation_basis(z_eval)
        coeffs_samples = self.coeffs_samples

        amplitude_samples = self.amplitude_samples
        nz_samples = (coeffs_samples @ basis_eval.T) * amplitude_samples[:, np.newaxis]

        nz_median = np.percentile(nz_samples, 50, axis=0)
        nz_mean = np.mean(nz_samples, axis=0)
        nz_std = np.std(nz_samples, axis=0)
        nz_lower = np.percentile(nz_samples, 16, axis=0)
        nz_upper = np.percentile(nz_samples, 84, axis=0)

        return {
            'z_eval': z_eval,
            'nz_median': nz_median,
            'nz_mean': nz_mean,
            'nz_std': nz_std,
            'nz_lower': nz_lower,
            'nz_upper': nz_upper,
            'basis_eval': basis_eval,
            'coeffs_samples': coeffs_samples,
            'amplitude_samples': amplitude_samples
        }
    
    def plot_fit(self, z_eval=None, n_eval_points=200, figsize=(9, 8), show_knots=True, show_integral_info=True, show_nnls=True):
        """
        Plot the data, fitted model, and uncertainty bands
        """
        if self.trace is None:
            raise ValueError("Model must be fitted before plotting. Call fit() first.")

        if z_eval is None:
            z_eval = np.linspace(self.zv.min(), self.zv.max(), n_eval_points)
        else:
            z_eval = np.asarray(z_eval)

        basis_eval = self._create_evaluation_basis(z_eval)
        coeffs_samples = self.coeffs_samples

        amplitude_samples = self.amplitude_samples
        nz_samples = (coeffs_samples @ basis_eval.T) * amplitude_samples[:, np.newaxis]

        nz_median = np.percentile(nz_samples, 50, axis=0)
        nz_mean = np.mean(nz_samples, axis=0)
        nz_std = np.std(nz_samples, axis=0)
        nz_lower = np.percentile(nz_samples, 16, axis=0)
        nz_upper = np.percentile(nz_samples, 84, axis=0)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.25)
        ax_main = fig.add_subplot(gs[0, :])
                
        info_text = f'Knots: {self.n_knots}\nDegree: {self.degree}\nBasis functions: {self.n_basis}'

        if show_nnls:
            coeffs_nnls, _ = self._get_initial_coeffs_and_amplitude(self.nz)
            nnls_pred_eval = basis_eval @ coeffs_nnls
            ax_main.plot(
                z_eval, 
                nnls_pred_eval, 
                color='lime', 
                linewidth=2, 
                linestyle=':', 
                label='NNLS fit', 
                alpha=0.8
                )
            if show_integral_info:
                nnls_integral = np.trapezoid(nnls_pred_eval, z_eval)
                info_text += f'\nNNLS integral: {nnls_integral:.3f}'

        ax_main.errorbar(
            self.zv, 
            self.nz, 
            yerr=self.nz_err,
            fmt='o', 
            color='black',
            alpha=0.7,
            capsize=3, 
            capthick=1, 
            label='Data'
            )

        if show_knots:
            knot_positions = self.knots[self.degree:-self.degree]
            for i, knot in enumerate(knot_positions):
                alpha_val = 0.6 if i == 0 else 0.4
                label_val = 'Knots' if i == 0 else None
                ax_main.axvline(knot, color='gray', linestyle='--', alpha=alpha_val, 
                          linewidth=1, label=label_val)

        ax_main.set_xlabel('Redshift (z)', fontsize=13)
        ax_main.set_ylabel('n(z)', fontsize=13)
        ax_main.legend(fontsize=11, loc='upper right', framealpha=0.9)
        ax_main.grid(True, alpha=0.3)
        
        ax_main.plot(
            z_eval, 
            nz_median, 
            color='red', 
            alpha=1,
            linewidth=2, 
            label='Bayesian median'
            )
        ax_main.fill_between(
            z_eval, 
            nz_lower, 
            nz_upper, 
            color='red', 
            alpha=0.3, 
            label='1σ'
            )

        ax_main.plot(
            z_eval, 
            nz_mean, 
            color='blue', 
            linestyle='--',
            linewidth=1, 
            label='mean'
            )
        ax_main.text(
            0.02, 
            0.98, 
            info_text, 
            transform=ax_main.transAxes,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), 
            fontsize=10
            )

        ax_basis = fig.add_subplot(gs[1, 0])
        n_show = self.n_basis
        indices = np.linspace(1, self.n_basis, n_show, dtype=int)
        colors = plt.cm.plasma(np.linspace(0, 1, n_show))
        
        for i, idx in enumerate(indices):
            basis_func = basis_eval[:, i]
            ax_basis.plot(
                z_eval, 
                basis_func, 
                color=colors[i], 
                alpha=0.7, 
                linewidth=1.5, 
                label=f'B_{idx}' if n_show <= 6 else None
                )
        
        ax_basis.set_xlabel('Redshift (z)', fontsize=12)
        ax_basis.set_ylabel('Basis amplitude', fontsize=12)
        ax_basis.set_title(f'B-spline Basis Functions (showing {n_show}/{self.n_basis})', fontsize=12)
        ax_basis.grid(True, alpha=0.3)
        if n_show <= 6:
            ax_basis.legend(fontsize=9, ncol=2)
        
        # Coefficient histogram (bottom right)
        ax_coeff = fig.add_subplot(gs[1, 1])
        coeff_means = np.mean(coeffs_samples, axis=0)
        coeff_stds = np.std(coeffs_samples, axis=0)
        coeff_medians = np.median(coeffs_samples, axis=0)
        x_pos = np.arange(self.n_basis)

        ax_coeff.bar(x_pos, coeff_means, yerr=coeff_stds, 
                           capsize=2, alpha=0.7, color='steelblue',
                           edgecolor='darkblue', linewidth=0.5, error_kw={'linewidth': 1})
        
        ax_coeff.scatter(x_pos, coeff_medians, color='red', s=15, alpha=0.8, 
                        zorder=3, label='Median')
        
        ax_coeff.set_xlabel('Basis Function Index', fontsize=12)
        ax_coeff.set_ylabel('Coefficient Value', fontsize=12)
        ax_coeff.set_title('Posterior Coefficient Distribution', fontsize=12)
        ax_coeff.grid(True, alpha=0.3, axis='y')
        ax_coeff.legend(fontsize=10)
        ax_coeff.set_xlim(-0.5, self.n_basis - 0.5)

        if self.n_basis > 15:
            tick_spacing = max(1, self.n_basis // 10)
            ax_coeff.set_xticks(x_pos[::tick_spacing])
            ax_coeff.set_xticklabels(x_pos[::tick_spacing])

        plt.tight_layout()
        return fig, (ax_main, ax_basis, ax_coeff)

    def predict(self, z_eval, return_samples=False, n_samples=None):
        """
        Once model is fitted, predict n(z) for new redshift values.
        """
        if self.trace is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        z_eval = np.asarray(z_eval)
        basis_eval = self._create_evaluation_basis(z_eval)

        coeffs_samples = self.coeffs_samples
        if n_samples is not None:
            coeffs_samples = coeffs_samples[:n_samples]
        amplitude_samples = self.amplitude_samples
        if n_samples is not None:
            amplitude_samples = amplitude_samples[:n_samples]
        nz_samples = (coeffs_samples @ basis_eval.T) * amplitude_samples[:, np.newaxis]

        if return_samples:
            return nz_samples
        else:
            return {
                'median': np.percentile(nz_samples, 50, axis=0),
                'mean': np.mean(nz_samples, axis=0),
                'std': np.std(nz_samples, axis=0),
                'lower_1sig': np.percentile(nz_samples, 16, axis=0),
                'upper_1sig': np.percentile(nz_samples, 84, axis=0)
            }

    def expect(self, z_eval, return_samples=False, n_samples=None):
        """
        Compute the expected value of n(z) for new redshift values.
        """
        if self.trace is None:
            raise ValueError("Model must be fitted before expectation. Call fit() first.")

        z_eval = np.asarray(z_eval)
        basis_eval = self._create_evaluation_basis(z_eval)

        coeffs_samples = self.coeffs_samples
        if n_samples is not None:
            coeffs_samples = coeffs_samples[:n_samples]
        amplitude_samples = self.amplitude_samples
        if n_samples is not None:
            amplitude_samples = amplitude_samples[:n_samples]
        
        nz_expectation = (coeffs_samples @ basis_eval.T) * amplitude_samples[:, np.newaxis]

        if return_samples:
            return nz_expectation
        else:
            return {
                'mean': np.mean(nz_expectation, axis=0),
                'std': np.std(nz_expectation, axis=0),
                'lower_1sig': np.percentile(nz_expectation, 16, axis=0),
                'upper_1sig': np.percentile(nz_expectation, 84, axis=0)
            }

    def _predict_normalized_pdf(self, z_eval, return_samples=False, n_samples=None):
        """Predict normalized n(z) as PDF with integral = 1."""
        if self.trace is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        z_eval = np.asarray(z_eval)
        basis_eval = self._create_evaluation_basis(z_eval)

        coeffs_samples = self.coeffs_samples
        if n_samples is not None:
            coeffs_samples = coeffs_samples[:n_samples]
        amplitude_samples = self.amplitude_samples
        if n_samples is not None:
            amplitude_samples = amplitude_samples[:n_samples]
            
        nz_samples = (coeffs_samples @ basis_eval.T) * amplitude_samples[:, np.newaxis]
        integrals = np.trapezoid(nz_samples, z_eval, axis=1)
        pdf_samples = nz_samples / integrals[:, np.newaxis]

        if return_samples:
            return pdf_samples
        else:
            median = np.percentile(pdf_samples, 50, axis=0)
            mean = np.mean(pdf_samples, axis=0)
            std = np.std(pdf_samples, axis=0)
            lower = np.percentile(pdf_samples, 16, axis=0)
            upper = np.percentile(pdf_samples, 84, axis=0)
            return median, mean, std, lower, upper

    def expectation(self, z_eval, n_samples=None):
        """Compute expectation value <z> of normalized PDF."""
        if self.trace is None:
            raise ValueError("Model must be fitted before computing expectation. Call fit() first.")

        z_eval = np.asarray(z_eval)
        pdf_samples = self._predict_normalized_pdf(z_eval, return_samples=True, n_samples=n_samples)
        
        expectation_samples = np.array([
            np.trapezoid(z_eval * pdf_sample, z_eval) 
            for pdf_sample in pdf_samples
        ])
        
        return np.mean(expectation_samples), np.std(expectation_samples), expectation_samples