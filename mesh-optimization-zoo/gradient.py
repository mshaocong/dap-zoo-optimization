import mfem.ser as mfem
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import griddata

def solve_poisson(mesh, order=1):
    """Solve Poisson equation on given mesh"""
    # Setup FE space
    fec = mfem.H1_FECollection(order, mesh.Dimension())
    fespace = mfem.FiniteElementSpace(mesh, fec)

    # Essential boundary conditions
    ess_tdof_list = mfem.intArray()
    ess_bdr = mfem.intArray([1]*mesh.bdr_attributes.Size())
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

    # Define coefficients
    alpha = mfem.ConstantCoefficient(1.0)
    rhs = mfem.ConstantCoefficient(1.0)

    # Setup bilinear and linear forms
    a = mfem.BilinearForm(fespace)
    a.AddDomainIntegrator(mfem.DiffusionIntegrator(alpha))
    a.Assemble()

    b = mfem.LinearForm(fespace)
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(rhs))
    b.Assemble()

    # Initialize solution
    x = mfem.GridFunction(fespace)
    x.Assign(0.0)

    # Form and solve system
    A = mfem.OperatorPtr()
    B = mfem.Vector()
    X = mfem.Vector()
    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)

    AA = mfem.OperatorHandle2SparseMatrix(A)
    M = mfem.GSSmoother(AA)
    mfem.PCG(AA, M, B, X, 1, 200, 1e-12, 0.0)
    a.RecoverFEMSolution(X, b, x)

    return x

def generate_perturbation(size, distribution='gaussian', amplitude=0.0001):
    """
    Generate perturbations according to specified distribution
    
    Args:
        size: Tuple of (num_vertices, dimensions)
        distribution: String specifying the distribution type
                     ('gaussian', 'uniform_sphere', 'rademacher', 'random_coordinate')
        amplitude: Scale factor for perturbations
        
    Returns:
        Array of perturbations with shape size
    """
    num_vertices, dims = size

    if distribution == 'gaussian':
        # Standard Gaussian distribution
        perturbation = amplitude * np.random.normal(loc=0.0, scale=1.0, size=size)
    
    elif distribution == 'uniform_sphere':
        # Uniform distribution over unit sphere
        perturbation = np.random.normal(0, 1, size=size)
        norms = np.linalg.norm(perturbation, axis=1, keepdims=True)
        perturbation = amplitude * perturbation / norms  * np.sqrt(num_vertices * dims)
        
    elif distribution == 'rademacher':
        # Rademacher distribution (random ±1)
        perturbation = amplitude * (2 * np.random.binomial(1, 0.5, size=size) - 1)  
    
    elif distribution == 'random_coordinate':
        # Randomly perturb one random vertex
        perturbation = np.zeros(size)
        vertex = np.random.randint(0, num_vertices)
        coord = np.random.randint(0, dims)
        perturbation[vertex, coord] = amplitude * np.sqrt(num_vertices * dims)
            
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
        
    return perturbation

def interpolate_solution(source_mesh, source_values, target_mesh):
    """
    Interpolate solution from source mesh to target mesh
    """
    source_vertices = np.vstack(source_mesh.GetVertexArray())
    target_vertices = np.vstack(target_mesh.GetVertexArray())
    
    interpolated_values = griddata(
        points=source_vertices,
        values=source_values,
        xi=target_vertices,
        method='linear'
    )
    
    return interpolated_values

def compare_solutions(fine_mesh, fine_values, coarse_mesh, coarse_values):
    """
    Compare solutions on different meshes by interpolating to the finer mesh
    """
    interpolated_coarse = interpolate_solution(coarse_mesh, coarse_values, fine_mesh)
    abs_diff = np.abs(fine_values - interpolated_coarse)
    rel_diff = abs_diff / (np.abs(fine_values) + 1e-10)  # Avoid division by zero
    rms_error = np.sqrt(np.mean(np.square(abs_diff)))
    
    # Compute various error metrics
    metrics = {
        'max_absolute_error': np.max(abs_diff),
        'mean_absolute_error': np.mean(abs_diff),
        'rms_error': rms_error,
        'max_relative_error': np.max(rel_diff),
        'mean_relative_error': np.mean(rel_diff)
    }
    
    return metrics

def estimate_gradient_batch(mesh, fine_mesh, fine_values, batch_size=10, 
                            perturbation_amplitude=0.0001, distribution='gaussian'):
    """
    Estimate gradient using batch sampling of random perturbations
    
    Args:
        mesh: Original mesh to optimize
        fine_mesh: Reference fine mesh
        fine_values: Solution values on fine mesh
        batch_size: Number of random perturbations to use
        perturbation_amplitude: Size of random perturbations
        distribution: Type of distribution to use ('gaussian', 'uniform_sphere', 'rademacher', 'random_coordinate', 'optimal')
        
    Returns:
        Estimated gradient
    """
    # Compute base solution and error
    base_solution = solve_poisson(mesh)
    base_values = base_solution.GetDataArray()
    base_metrics = compare_solutions(fine_mesh, fine_values, mesh, base_values)
    base_error = base_metrics['rms_error']

    # Get mesh vertices
    vertices = np.vstack(mesh.GetVertexArray())
    num_vertices = vertices.shape[0]
    dims = vertices.shape[1]

    # Initialize gradient accumulator
    accumulated_gradient = np.zeros((num_vertices, dims), dtype=float)

    if distribution != 'optimal':
        for _ in range(batch_size):
            # Generate random perturbation
            perturbation = generate_perturbation(size=(num_vertices, dims), distribution=distribution, amplitude=perturbation_amplitude)
            # Set perturbation to zero on boundary vertices
            for i in range(num_vertices):
                if vertices[i,0] in [0, 1] or vertices[i,1] in [0, 1]:
                    perturbation[i, :] = 0.0
            # Create perturbed mesh
            perturbed_mesh = mfem.Mesh(mesh)
            perturbed_vertices = perturbed_mesh.GetVertexArray()
            for i in range(num_vertices):
                perturbed_vertices[i][0] = vertices[i,0] + perturbation[i,0]
                perturbed_vertices[i][1] = vertices[i,1] + perturbation[i,1]
            # Solve on perturbed mesh and compute error
            perturbed_solution = solve_poisson(perturbed_mesh)
            perturbed_values = perturbed_solution.GetDataArray()
            perturbed_metrics = compare_solutions(fine_mesh, fine_values, perturbed_mesh, perturbed_values)
            perturbed_error = perturbed_metrics['rms_error']
            # Accumulate gradient estimate
            error_diff = perturbed_error - base_error
            accumulated_gradient += (error_diff / (perturbation_amplitude ** 2)) * perturbation
        # Average the gradients
        gradient = accumulated_gradient / batch_size
    else:
        # Optimal method
        # First compute an initial gradient using 'uniform_sphere' distribution
        for _ in range(batch_size):
            perturbation = generate_perturbation(size=(num_vertices, dims), distribution='uniform_sphere', amplitude=perturbation_amplitude)
            for i in range(num_vertices):
                if vertices[i,0] in [0, 1] or vertices[i,1] in [0, 1]:
                    perturbation[i, :] = 0.0
            # Create perturbed mesh
            perturbed_mesh = mfem.Mesh(mesh)
            perturbed_vertices = perturbed_mesh.GetVertexArray()
            for i in range(num_vertices):
                perturbed_vertices[i][0] = vertices[i,0] + perturbation[i,0]
                perturbed_vertices[i][1] = vertices[i,1] + perturbation[i,1]
            # Solve on perturbed mesh and compute error
            perturbed_solution = solve_poisson(perturbed_mesh)
            perturbed_values = perturbed_solution.GetDataArray()
            perturbed_metrics = compare_solutions(fine_mesh, fine_values, perturbed_mesh, perturbed_values)
            perturbed_error = perturbed_metrics['rms_error']
            # Accumulate gradient estimate
            error_diff = perturbed_error - base_error
            accumulated_gradient += (error_diff / (perturbation_amplitude ** 2)) * perturbation
        # Average the gradients
        initial_gradient = accumulated_gradient / batch_size
        
        # Now do the 'optimal' method adjustments
        grad_norm_square = np.sum(initial_gradient.flatten() ** 2)
        drift = initial_gradient.flatten() / np.sqrt(grad_norm_square) 
        
        accumulated_gradient = np.zeros((num_vertices, dims), dtype=float)
        for _ in range(batch_size):
            perturbation = np.random.normal(0, 1, size=(num_vertices, dims))
            norms = np.linalg.norm(perturbation, axis=1, keepdims=True)
            z = perturbation / norms * np.sqrt(num_vertices * dims)
            ratio = np.dot(initial_gradient.flatten(), z.flatten()) / grad_norm_square if grad_norm_square > 0 else 0
            xi = np.random.choice([-1, 1])
            corrected_z = z - ratio * initial_gradient + xi * drift.reshape((num_vertices, dims))
            perturbation = perturbation_amplitude * corrected_z
            # Set perturbation to zero on boundary vertices
            for i in range(num_vertices):
                if vertices[i,0] in [0, 1] or vertices[i,1] in [0, 1]:
                    perturbation[i, :] = 0.0
            # Create perturbed mesh
            perturbed_mesh = mfem.Mesh(mesh)
            perturbed_vertices = perturbed_mesh.GetVertexArray()
            for i in range(num_vertices):
                perturbed_vertices[i][0] = vertices[i,0] + perturbation[i,0]
                perturbed_vertices[i][1] = vertices[i,1] + perturbation[i,1]
            # Solve on perturbed mesh and compute error
            perturbed_solution = solve_poisson(perturbed_mesh)
            perturbed_values = perturbed_solution.GetDataArray()
            perturbed_metrics = compare_solutions(fine_mesh, fine_values, perturbed_mesh, perturbed_values)
            perturbed_error = perturbed_metrics['rms_error']
            # Accumulate gradient estimate
            error_diff = perturbed_error - base_error
            accumulated_gradient += (error_diff / (perturbation_amplitude ** 2)) * perturbation
        # Combine initial gradient and new accumulated gradient
        gradient = (initial_gradient + accumulated_gradient / batch_size) / 2

    return gradient

def update_mesh(mesh, gradient, learning_rate=0.1):
    """
    Update the mesh vertices according to the negative gradient
    
    Args:
        mesh: The mesh to update
        gradient: The gradient of the error with respect to the mesh vertices
        learning_rate: The step size for the gradient update
        
    Returns:
        The updated mesh
    """
    updated_mesh = mfem.Mesh(mesh)
    vertices = updated_mesh.GetVertexArray()
    vertices = np.vstack(vertices)
    perturbation = learning_rate * gradient

    for i in range(vertices.shape[0]):
        # Check if it's an interior vertex
        if not (vertices[i, 0] in [0, 1] or vertices[i, 1] in [0, 1]):
            vertices[i] -= perturbation[i]
        else:
            perturbation[i, :] = 0.0  # No perturbation on boundary vertices

    # Update the mesh vertices
    vertices_handle = updated_mesh.GetVertexArray()
    for i in range(vertices.shape[0]):
        vertices_handle[i][0] = vertices[i,0]
        vertices_handle[i][1] = vertices[i,1]

    return updated_mesh

def plot_gradient(mesh, gradient_magnitude, ax=None, vmin=None, vmax=None, title='Solution Gradient Magnitude'):
    """Plot gradient magnitude with colorbar"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        fig = None
    vertices = mesh.GetVertexArray()
    vertices = np.vstack(vertices)
    triang = tri.Triangulation(vertices[:,0], vertices[:,1])

    ax.set_aspect('equal')
    im = ax.tripcolor(triang, gradient_magnitude, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return fig, ax, im

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create meshes
    fine_mesh = mfem.Mesh(20, 20, "TRIANGLE")
    coarse_mesh = mfem.Mesh(10, 10, "TRIANGLE")

    # Solve on fine mesh
    fine_solution = solve_poisson(fine_mesh)
    fine_values = fine_solution.GetDataArray()

    # Solve on coarse mesh
    coarse_solution = solve_poisson(coarse_mesh)
    coarse_values = coarse_solution.GetDataArray()

    # Compute error before any mesh update
    initial_metrics = compare_solutions(fine_mesh, fine_values, coarse_mesh, coarse_values)
    print(f"Initial RMS Error: {initial_metrics['rms_error']:.6f}")

    # Compute ground truth gradient
    print("\nComputing ground truth gradient with 'uniform_sphere' distribution and batch_size=10000")
    gradient_gt = estimate_gradient_batch(coarse_mesh, fine_mesh, fine_values, batch_size=10000, perturbation_amplitude=0.0001, distribution='uniform_sphere')
    gradient_gt_magnitude = np.linalg.norm(gradient_gt, axis=1)

    # Initialize dictionaries to store gradients
    gradient_magnitudes = {}

    # List of gradient estimation methods
    distributions = ['uniform_sphere', 'optimal']

    for dist in distributions:
        print(f"\nEstimating gradient using distribution: {dist}")
        # Estimate gradient
        gradient = estimate_gradient_batch(coarse_mesh, fine_mesh, fine_values, batch_size=128, perturbation_amplitude=0.0001, distribution=dist)

        # Compute gradient magnitude
        gradient_magnitude = np.linalg.norm(gradient, axis=1)
        gradient_magnitudes[dist] = gradient_magnitude

    # Compute common color scale limits for gradient magnitudes
    all_gradient_magnitudes = [gradient_gt_magnitude, gradient_magnitudes['uniform_sphere'], gradient_magnitudes['optimal']]
    gradient_magnitude_min = 0.4 # min([gm.min() for gm in all_gradient_magnitudes])
    gradient_magnitude_max = max([gm.max() for gm in all_gradient_magnitudes])

    # Plot gradient magnitudes of 'ground truth', 'uniform_sphere', and 'optimal' on the same figure

    fig, axes = plt.subplots(1, 3, figsize=(18,6))
    images = []

    # Plot ground truth gradient magnitude
    gradient_magnitude = gradient_gt_magnitude
    _, _, im = plot_gradient(coarse_mesh, gradient_magnitude, ax=axes[0], vmin=gradient_magnitude_min, vmax=gradient_magnitude_max, title='Ground Truth')
    images.append(im)

    # Plot 'uniform_sphere' gradient magnitude
    gradient_magnitude = gradient_magnitudes['uniform_sphere']
    _, _, im = plot_gradient(coarse_mesh, gradient_magnitude, ax=axes[1], vmin=gradient_magnitude_min, vmax=gradient_magnitude_max, title='Uniform Perturbation')
    images.append(im)

    # Plot 'optimal' gradient magnitude
    gradient_magnitude = gradient_magnitudes['optimal']
    _, _, im = plot_gradient(coarse_mesh, gradient_magnitude, ax=axes[2], vmin=gradient_magnitude_min, vmax=gradient_magnitude_max, title='DAP (Estimated Gradient)')
    images.append(im)

    # Adjust layout
    plt.tight_layout()

    # Create a single colorbar shared among all subplots
    cbar = fig.colorbar(images[0], ax=axes.ravel().tolist(), shrink=0.6)
    # cbar.set_label('Gradient Magnitude')

    # Save the figure
    plt.savefig("gradient_magnitudes_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()