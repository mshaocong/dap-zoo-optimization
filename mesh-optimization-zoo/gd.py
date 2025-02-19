import mfem.ser as mfem
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri 
import mfem.ser as mfem
import numpy as np
from scipy.interpolate import griddata 
import argparse
import wandb 

def generate_perturbation(size, distribution, amplitude=0.0001):
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
    
    elif distribution == 'uniform_sphere' or 'optimal':
        # Uniform distribution over unit sphere
        # Generate random directions
        perturbation = np.random.normal(0, 1, size=size)
        # Normalize each vertex perturbation to unit length
        norms = np.linalg.norm(perturbation, axis=1, keepdims=True)
        perturbation = amplitude * perturbation / norms  * np.sqrt(num_vertices * dims)
        
    elif distribution == 'rademacher':
        # Rademacher distribution (random ±1)
        perturbation = amplitude * (2 * np.random.binomial(1, 0.5, size=size) - 1)  
        
    elif distribution == 'random_coordinate':
        # Randomly perturb one random vertex
        perturbation = np.zeros(size)
        # Pick random vertex and coordinate
        vertex = np.random.randint(0, num_vertices)
        coord = np.random.randint(0, dims)
        # Set random ±amplitude for chosen vertex's coordinate 
        perturbation[vertex, coord] = amplitude * np.sqrt(num_vertices * dims)
            
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
        
    return perturbation

def update_mesh(mesh, gradient, learning_rate=0.1): 
    perturbed_mesh = mfem.Mesh(mesh) 
    # Get vertices array
    vertices = perturbed_mesh.GetVertexArray()
    vertices = np.vstack(vertices) 
    # Generate random perturbations 
    perturbation = learning_rate * gradient
    
    # Apply perturbations to interior vertices
    for i in range(vertices.shape[0]):
        # Check if it's an interior vertex
        if not (vertices[i,0] in [0, 1] or vertices[i,1] in [0, 1]):
            vertices[i] -= perturbation[i]
    
    # Update the mesh vertices
    vertices_handle = perturbed_mesh.GetVertexArray()
    for i in range(vertices.shape[0]):
        vertices_handle[i][0] = vertices[i,0]
        vertices_handle[i][1] = vertices[i,1] 

    return perturbed_mesh, perturbation


def interpolate_solution(source_mesh, source_values, target_mesh):
    """
    Interpolate solution from source mesh to target mesh
    
    Args:
        source_mesh: MFEM mesh containing source solution
        source_values: Solution values on source mesh
        target_mesh: MFEM mesh to interpolate solution onto
        
    Returns:
        Interpolated values on target mesh points
    """
    # Get vertex coordinates from both meshes
    source_vertices = np.vstack(source_mesh.GetVertexArray())
    target_vertices = np.vstack(target_mesh.GetVertexArray())
    
    # Interpolate values from source to target mesh points
    interpolated_values = griddata(
        points=source_vertices,
        values=source_values,
        xi=target_vertices,
        method='linear'
    )
    
    return interpolated_values

def compare_solutions(fine_mesh, fine_values, perturbed_mesh, perturbed_values):
    """
    Compare solutions on different meshes by interpolating to the finer mesh
    
    Args:
        fine_mesh: Fine resolution mesh
        fine_values: Solution values on fine mesh
        perturbed_mesh: Perturbed mesh
        perturbed_values: Solution values on perturbed mesh
        
    Returns:
        Dictionary containing different error metrics
    """
    # Interpolate perturbed solution onto fine mesh
    interpolated_perturbed = interpolate_solution(
        perturbed_mesh, 
        perturbed_values,
        fine_mesh
    )
    
    # Calculate differences
    abs_diff = np.abs(fine_values - interpolated_perturbed)
    rel_diff = abs_diff / (np.abs(fine_values) + 1e-10)  # Add small constant to avoid division by zero
    
    # Compute various error metrics
    metrics = {
        'max_absolute_error': np.max(abs_diff),
        'mean_absolute_error': np.mean(abs_diff),
        'rms_error': np.sqrt(np.mean(np.square(abs_diff))),
        'max_relative_error': np.max(rel_diff),
        'mean_relative_error': np.mean(rel_diff)
    }
    
    return metrics, abs_diff, interpolated_perturbed


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
def perturb_mesh(original_mesh, distribution='gaussian', amplitude=0.0001):
    """
    Perturb vertex positions while maintaining mesh connectivity
    
    Args:
        original_mesh: Original MFEM mesh
        distribution: Type of perturbation distribution
                     ('gaussian', 'uniform_sphere', 'rademacher', 'random_coordinate')
        amplitude: Scale factor for perturbations
        
    Returns:
        Tuple of (perturbed mesh, perturbation array)
    """
    # Create a deep copy of the original mesh
    perturbed_mesh = mfem.Mesh(original_mesh) 
    
    # Get vertices array
    vertices = perturbed_mesh.GetVertexArray()
    vertices = np.vstack(vertices) 
    
    # Generate random perturbations according to specified distribution
    perturbation = generate_perturbation(
        size=(vertices.shape[0], 2),
        distribution=distribution,
        amplitude=amplitude
    )
    
    # Apply perturbations to interior vertices
    for i in range(vertices.shape[0]):
        # Check if it's an interior vertex
        if not (vertices[i,0] in [0, 1] or vertices[i,1] in [0, 1]):
            vertices[i] += perturbation[i]
    
    # Update the mesh vertices
    vertices_handle = perturbed_mesh.GetVertexArray()
    for i in range(vertices.shape[0]):
        vertices_handle[i][0] = vertices[i,0]
        vertices_handle[i][1] = vertices[i,1] 

    return perturbed_mesh, perturbation

def plot_raw_solution(mesh, values, ax):
    """Plot raw solution without any decorative elements"""
    vertices = mesh.GetVertexArray()
    vertices = np.vstack(vertices)
    triang = tri.Triangulation(vertices[:,0], vertices[:,1])
    
    ax.set_aspect('equal')
    ax.tripcolor(triang, values, shading='gouraud', cmap='viridis')
    
    # Remove all decorative elements
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def plot_gradient(mesh, gradient_values):
    """Plot gradient with colorbar"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    vertices = mesh.GetVertexArray()
    vertices = np.vstack(vertices)
    triang = tri.Triangulation(vertices[:,0], vertices[:,1])
    
    ax.set_aspect('equal')
    im = ax.tripcolor(triang, gradient_values, shading='gouraud', cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Gradient Magnitude')
    
    # Remove axis ticks but keep the frame
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.title('Solution Gradient Magnitude')
    return fig, ax
import mfem.ser as mfem
import numpy as np
from scipy.interpolate import griddata

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
        
    Returns:
        Estimated gradient
    """
    # Get baseline error
    base_solution = solve_poisson(mesh)
    base_values = base_solution.GetDataArray()
    base_metrics, _, _ = compare_solutions(fine_mesh, fine_values, mesh, base_values)
    base_error = base_metrics['rms_error']
    
    # Initialize gradient accumulator
    vertices = np.vstack(mesh.GetVertexArray())
    
    # Generate multiple random perturbations 
    if distribution != 'optimal':
        accumulated_gradient = np.zeros_like(vertices, dtype=float)
    
        for i in range(batch_size):
            # Generate random perturbation
            perturbation = generate_perturbation(
                size=(vertices.shape[0], 2),
                distribution=distribution,
                amplitude=perturbation_amplitude
            )
            # Create perturbed mesh
            perturbed_mesh = mfem.Mesh(mesh)
            perturbed_vertices = perturbed_mesh.GetVertexArray()
            
            # Apply perturbation to interior vertices only
            for j in range(vertices.shape[0]):
                if not (vertices[j,0] in [0, 1] or vertices[j,1] in [0, 1]):
                    perturbed_vertices[j][0] = vertices[j,0] + perturbation[j,0]
                    perturbed_vertices[j][1] = vertices[j,1] + perturbation[j,1]
            
            # Solve on perturbed mesh
            perturbed_solution = solve_poisson(perturbed_mesh)
            perturbed_values = perturbed_solution.GetDataArray()
            
            # Calculate error on perturbed mesh
            metrics, _, _ = compare_solutions(
                fine_mesh, 
                fine_values, 
                perturbed_mesh, 
                perturbed_values
            )
            perturbed_error = metrics['rms_error']
            
            # Accumulate gradient estimate
            error_diff = perturbed_error - base_error
            accumulated_gradient += (error_diff / perturbation_amplitude) * perturbation
        # Average the gradients
        final_gradient = accumulated_gradient / batch_size
    else: 
        accumulated_gradient = np.zeros_like(vertices, dtype=float)
    
        for i in range(batch_size):
            # Generate random perturbation
            perturbation = generate_perturbation(
                size=(vertices.shape[0], 2),
                distribution="uniform_sphere",
                amplitude=perturbation_amplitude
            )
            # Create perturbed mesh
            perturbed_mesh = mfem.Mesh(mesh)
            perturbed_vertices = perturbed_mesh.GetVertexArray()
            
            # Apply perturbation to interior vertices only
            for j in range(vertices.shape[0]):
                if not (vertices[j,0] in [0, 1] or vertices[j,1] in [0, 1]):
                    perturbed_vertices[j][0] = vertices[j,0] + perturbation[j,0]
                    perturbed_vertices[j][1] = vertices[j,1] + perturbation[j,1]
            
            # Solve on perturbed mesh
            perturbed_solution = solve_poisson(perturbed_mesh)
            perturbed_values = perturbed_solution.GetDataArray()
            
            # Calculate error on perturbed mesh
            metrics, _, _ = compare_solutions(
                fine_mesh, 
                fine_values, 
                perturbed_mesh, 
                perturbed_values
            )
            perturbed_error = metrics['rms_error']
            
            # Accumulate gradient estimate
            error_diff = perturbed_error - base_error
            accumulated_gradient += (error_diff / perturbation_amplitude) * perturbation
        # Average the gradients
        final_gradient = accumulated_gradient / batch_size
        accumulated_gradient = np.zeros_like(vertices, dtype=float)
    
        for i in range(batch_size):
            perturbation = np.random.normal(0, 1, size=(vertices.shape[0], 2) )
            # Normalize each vertex perturbation to unit length
            norms = np.linalg.norm(perturbation, axis=1, keepdims=True)
            z = perturbation / norms  * np.sqrt(vertices.shape[0] * 2)
            grad_norm_square = np.sum(final_gradient.flatten() ** 2)
            if grad_norm_square > 0.001: 
                drift = final_gradient.flatten() / np.sqrt(grad_norm_square) 
                ratio = np.dot(final_gradient.flatten(), z.flatten()) / grad_norm_square 
                xi = np.random.randint(0, 2, (1,) ) * 2 - 1   
                z = z - ratio * final_gradient + xi * drift.reshape(final_gradient.shape) 
            perturbation = perturbation_amplitude * z  
            # Create perturbed mesh
            perturbed_mesh = mfem.Mesh(mesh)
            perturbed_vertices = perturbed_mesh.GetVertexArray()
            
            # Apply perturbation to interior vertices only
            for j in range(vertices.shape[0]):
                if not (vertices[j,0] in [0, 1] or vertices[j,1] in [0, 1]):
                    perturbed_vertices[j][0] = vertices[j,0] + perturbation[j,0]
                    perturbed_vertices[j][1] = vertices[j,1] + perturbation[j,1]
            
            # Solve on perturbed mesh
            perturbed_solution = solve_poisson(perturbed_mesh)
            perturbed_values = perturbed_solution.GetDataArray()
            
            # Calculate error on perturbed mesh
            metrics, _, _ = compare_solutions(
                fine_mesh, 
                fine_values, 
                perturbed_mesh, 
                perturbed_values
            )
            perturbed_error = metrics['rms_error']
            
            # Accumulate gradient estimate
            error_diff = perturbed_error - base_error
            accumulated_gradient += (error_diff / perturbation_amplitude) * perturbation
        # Average the gradients
        final_gradient = (accumulated_gradient / batch_size + final_gradient)/2
    return final_gradient

def parse_args():
    parser = argparse.ArgumentParser(description='Mesh Optimization Parameters')
    
    # Mesh parameters
    parser.add_argument('--fine_mesh_size', type=int, default=20,
                        help='Size of the fine mesh grid')
    parser.add_argument('--coarse_mesh_size', type=int, default=10,
                        help='Size of the coarse mesh grid')
    
    # Optimization parameters
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='Number of optimization iterations')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of samples per gradient estimation')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for gradient descent')
    parser.add_argument('--perturbation_amplitude', type=float, default=0.00001,
                        help='Size of random perturbations')
    
    # Wandb parameters
    parser.add_argument('--wandb_project', type=str, default='mesh-optimization',
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='WandB entity name')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[],
                        help='Tags for the WandB run')
    parser.add_argument('--perturbation_distribution', type=str, 
                    choices=['gaussian', 'uniform_sphere', 'rademacher', 'random_coordinate', 'optimal'],
                    default='gaussian',
                    help='Type of perturbation distribution')
    return parser.parse_args()

def optimize_mesh(mesh, fine_mesh, fine_values, num_iterations=10, 
                 batch_size=10, learning_rate=0.01, 
                 perturbation_amplitude=0.0001, distribution='gaussian'):
    """
    Optimize mesh using batch gradient descent with WandB logging
    """
    current_mesh = mfem.Mesh(mesh)
    error_history = []
    
    # Log mesh properties
    wandb.log({
        "initial_vertices": len(np.vstack(current_mesh.GetVertexArray())),
        "mesh_dimension": current_mesh.Dimension()
    })
    
    for iteration in range(num_iterations):
        # Estimate gradient using batch sampling
        gradient = estimate_gradient_batch(
            current_mesh, 
            fine_mesh, 
            fine_values, 
            batch_size=batch_size,
            perturbation_amplitude=perturbation_amplitude,
            distribution=distribution
        )
        
        # Update mesh using gradient
        current_mesh, perturbation = update_mesh(current_mesh, gradient, learning_rate)
        
        # Calculate and store error
        solution = solve_poisson(current_mesh)
        values = solution.GetDataArray()
        metrics, abs_diff, interpolated = compare_solutions(
            fine_mesh, 
            fine_values, 
            current_mesh, 
            values
        )
        error_history.append(metrics['rms_error'])
        
        # Log metrics to WandB
        wandb.log({
            "iteration": iteration + 1,
            "rms_error": metrics['rms_error'],
            "max_absolute_error": metrics['max_absolute_error'],
            "mean_absolute_error": metrics['mean_absolute_error'],
            "max_relative_error": metrics['max_relative_error'],
            "mean_relative_error": metrics['mean_relative_error'],
            "gradient_norm": np.linalg.norm(gradient),
            "perturbation_norm": np.linalg.norm(perturbation)
        })
        
        # Every 10 iterations, log solution visualization
        if (iteration + 1) % 10 == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_raw_solution(current_mesh, values, ax)
            wandb.log({f"solution_viz": wandb.Image(fig)})
            plt.close(fig)
            
            # Log gradient visualization
            fig, _ = plot_gradient(current_mesh, np.linalg.norm(gradient, axis=1))
            wandb.log({f"gradient_viz": wandb.Image(fig)})
            plt.close(fig)
        
        print(f"Iteration {iteration + 1}, RMS Error: {error_history[-1]:.6f}")
    
    return current_mesh, error_history

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        tags=args.wandb_tags,
        config=vars(args)
    )
    
    # Create meshes
    fine_mesh = mfem.Mesh(args.fine_mesh_size, args.fine_mesh_size, "TRIANGLE")
    coarse_mesh = mfem.Mesh(args.coarse_mesh_size, args.coarse_mesh_size, "TRIANGLE")
    
    # Solve on fine mesh to get reference solution
    fine_solution = solve_poisson(fine_mesh)
    fine_values = fine_solution.GetDataArray()
    
    # Log initial solution visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_raw_solution(fine_mesh, fine_values, ax)
    wandb.log({"reference_solution": wandb.Image(fig)})
    plt.close(fig)
    
    # Optimize mesh
    optimized_mesh, error_history = optimize_mesh(
        coarse_mesh,
        fine_mesh,
        fine_values,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        perturbation_amplitude=args.perturbation_amplitude,
        distribution=args.perturbation_distribution
    )
    
    # Log final error history plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(error_history) + 1), error_history)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('RMS Error')
    ax.set_title('Optimization Error History')
    wandb.log({"error_history": wandb.Image(fig)})
    plt.close(fig)
    
    # Log final solution
    final_solution = solve_poisson(optimized_mesh)
    final_values = final_solution.GetDataArray()
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_raw_solution(optimized_mesh, final_values, ax)
    wandb.log({"final_solution": wandb.Image(fig)})
    plt.close(fig)
    
    wandb.finish()

if __name__ == "__main__":
    main()