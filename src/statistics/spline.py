import numpy as np
from scipy.optimize import minimize, differential_evolution, nnls
from scipy.interpolate import BSpline

def mspline_basis(x, knots, degree=3):
    """
    Create M-spline basis functions.
    """
    # Create B-spline basis
    n_basis = len(knots) - degree - 1
    
    if n_basis <= 0:
        raise ValueError(f"Not enough knots. Got {len(knots)} knots for degree {degree}, need at least {degree + 2}")
    
    basis_functions = []
    
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0

        bspline = BSpline(knots, coeffs, degree)
        
        basis_vals = bspline(x, extrapolate=False)
        basis_vals = np.nan_to_num(basis_vals, nan=0.0, posinf=0.0, neginf=0.0)

        if np.any(basis_vals < 0):
            raise ValueError("M-spline basis function values must be non-negative.")
        
        basis_functions.append(basis_vals)
    
    return np.array(basis_functions).T

def mspline_pdf(x, coeffs, knots, degree=3):
    """
    Evaluate M-spline PDF given coefficients.
    """
    basis_matrix = mspline_basis(x, knots, degree)
    pdf = np.dot(basis_matrix, coeffs)
    return pdf

def optimize_knot_positions(x_data, y_data, y_err, n_knots=12, method='differential_evolution'):
    """
    Directly optimize knot positions to minimize fitting error
    """
    x_min = min(x_data)
    x_max = max(x_data)

    def objective(knot_positions):
        # Ensure knots are sorted and within bounds
        knots_sorted = np.sort(knot_positions)
        knots_sorted = np.clip(knots_sorted, x_min + 1e-6, x_max - 1e-6)
        
        try:
            # Create full knot vector
            full_knots = np.concatenate([
                [x_min] * 4, knots_sorted, [x_max] * 4
            ])
            
            # Fit M-spline
            basis_matrix = mspline_basis(x_data, full_knots, degree=3)
            weights = 1.0 / (y_err**2)
            
            A = basis_matrix * np.sqrt(weights)[:, np.newaxis]
            b = y_data * np.sqrt(weights)
            
            coeffs, _ = nnls(A, b)
            
            # Compute weighted chi-squared
            y_pred = basis_matrix @ coeffs
            chi2 = np.sum(((y_data - y_pred) / y_err)**2)
            
            # Add penalty for knots that are too close together
            min_spacing = (x_max - x_min) / (n_knots * 5)  # Minimum spacing
            spacing_penalty = 0
            for i in range(len(knots_sorted)-1):
                if (knots_sorted[i+1] - knots_sorted[i]) < min_spacing:
                    spacing_penalty += 1e4 * (min_spacing - (knots_sorted[i+1] - knots_sorted[i]))**2

            return chi2 + spacing_penalty
            
        except:
            return 1e10  # Large penalty for invalid configurations
    
    # Initial guess: uniform spacing
    init_knots = np.linspace(x_min, x_max, n_knots + 2)[1:-1]
    # Bounds: each knot must be within the data range
    bounds = [(x_min + 1e-6, x_max - 1e-6) for _ in range(n_knots)]
    
    if method == 'differential_evolution':
        # Global optimization
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            tol=0.05,
            disp=False,
            polish=True,     
        )
        
        # Check if optimization "failed" but still found a reasonable solution
        if not result.success:
            print(f"   DE didn't fully converge (reason: {result.message})")
                
    else:
        # Local optimization
        result = minimize(objective, init_knots, bounds=bounds, method='L-BFGS-B')
        
        if not result.success:
            print(f"   L-BFGS-B didn't converge (reason: {result.message})")
            print(f"   Found solution with score: {result.fun:.3f}")
    
    knots = np.sort(result.x)
    return knots, result

def optimize_knots(x_data, y_data, y_err, max_knots=16, method='differential_evolution'):
    """
    Optimize knot positions and number of knots for M-spline fitting.
    Returns the best result with respect to scoring function.
    """
    results = []
    
    for n_knots in range(4, max_knots + 1):
        print(f"Optimizing with {n_knots} knots using {method}...")
        try:
            knots, result = optimize_knot_positions(x_data, y_data, y_err, n_knots=n_knots, method=method)
            results.append({
                'n_knots': n_knots,
                'knots': knots,
                'score': result.fun,
                'result': result,
                'success': result.success,
                'message': getattr(result, 'message', ''),
                'nfev': getattr(result, 'nfev', 0)
            })
            
        except Exception as e:
            print(f"Failed with {n_knots} knots: {e}")
            continue
    
    return results

def adaptive_residual_refinement(x_data, y_data, y_err, max_knots=20, target_chi2_red=.5):
    """
    Iterative refinement: start simple, then add knots where residuals are largest
    """
    x_min, x_max = min(x_data), max(x_data)
    results = []
    
    # Start with minimal knots (just endpoints + a few internal)
    current_knots = np.linspace(x_min, x_max, 6)[1:-1]  # Start with 4 internal knots
    
    for iteration in range(max_knots - 4):
        try:
            # Fit current model
            full_knots = np.concatenate([[x_min] * 4, current_knots, [x_max] * 4])
            basis_matrix = mspline_basis(x_data, full_knots, degree=3)
            
            weights = 1.0 / (y_err**2)
            A = basis_matrix * np.sqrt(weights)[:, np.newaxis]
            b = y_data * np.sqrt(weights)
            
            coeffs, _ = nnls(A, b)
            y_pred = basis_matrix @ coeffs
            
            # Compute residuals and fit quality
            residuals = (y_data - y_pred) / y_err
            chi2 = np.sum(residuals**2)
            dof = len(y_data) - len(coeffs)
            chi2_red = chi2 / dof if dof > 0 else np.inf
            
            results.append({
                'n_knots': len(current_knots),
                'knots': current_knots.copy(),
                'chi2_red': chi2_red,
                'coeffs': coeffs,
                'full_knots': full_knots,
                'prediction': y_pred.copy()
            })
            
            # Check if we've reached target quality
            if chi2_red <= target_chi2_red:
                break
                
            # Find where to add next knot (largest residual)
            abs_residuals = np.abs(residuals)
            max_residual_idx = np.argmax(abs_residuals)
            new_knot_position = x_data[max_residual_idx]
            
            # Make sure new knot isn't too close to existing ones
            min_distance = (x_max - x_min) / (max_knots * 3)
            distances = np.abs(current_knots - new_knot_position)
            
            if np.min(distances) < min_distance:
                # Find alternative position near the worst residual
                residual_region = abs_residuals > np.percentile(abs_residuals, 80)
                candidate_positions = x_data[residual_region]
                
                for pos in candidate_positions:
                    distances = np.abs(current_knots - pos)
                    if np.min(distances) >= min_distance:
                        new_knot_position = pos
                        break
                else:
                    # No good position found, stop refinement
                    break
            
            # Add the new knot
            current_knots = np.sort(np.append(current_knots, new_knot_position))
            
        except Exception as e:
            print(f"   Residual refinement failed at iteration {iteration}: {e}")
            break
    
    return results

def adaptive_refinement_knots(x_data, y_data, y_err, max_knots=20, target_chi2_red=1.0):
    """
    Curvature-based adaptive refinement: place knots where data has high curvature
    """
    x_min = min(x_data)
    x_max = max(x_data)
    results = []
    
    def get_curvature_knots(n_knots):
        """Place knots where data has high curvature (second derivative)."""
        if len(x_data) < 5:
            return np.linspace(x_min, x_max, n_knots + 2)[1:-1]
        
        # approximate second derivative
        y_smooth = np.convolve(y_data, np.array([1, -2, 1]), mode='valid')
        x_smooth = x_data[1:-1]

        weights = np.abs(y_smooth) / np.interp(x_smooth, x_data, y_err**2)
        uniform_knots = np.linspace(x_min, x_max, max(n_knots//2, 2))

        if len(weights) > 0 and n_knots > len(uniform_knots):
            probabilities = weights / np.sum(weights)
            curvature_knots = np.random.choice(x_smooth, 
                                             size=n_knots - len(uniform_knots), 
                                             p=probabilities, replace=False)
            all_knots = np.concatenate([uniform_knots, curvature_knots])
        else:
            all_knots = uniform_knots
        
        return np.sort(all_knots[(all_knots > x_min) & (all_knots < x_max)])[:n_knots]

    for n_knots in range(4, max_knots + 1):
        knots = get_curvature_knots(n_knots)
        try:
            full_knots = np.concatenate([[x_min] * 4, knots, [x_max] * 4])
            basis_matrix = mspline_basis(x_data, full_knots, degree=3)
            weights = 1.0 / (y_err**2)
            A = basis_matrix * np.sqrt(weights)[:, np.newaxis]
            b = y_data * np.sqrt(weights)
            coeffs, _ = nnls(A, b)
            y_pred = basis_matrix @ coeffs
            residuals = (y_data - y_pred) / y_err
            chi2 = np.sum(residuals**2)
            dof = len(y_data) - len(coeffs)
            chi2_red = chi2 / dof if dof > 0 else np.inf
            
            results.append({
                'n_knots': n_knots,
                'knots': knots,
                'chi2_red': chi2_red,
                'coeffs': coeffs,
                'full_knots': full_knots,
                'prediction': y_pred
            })
            
            if chi2_red <= target_chi2_red:
                break
                
        except Exception as e:
            print(f"   Failed with {n_knots} knots: {e}")
            continue
    
    return results