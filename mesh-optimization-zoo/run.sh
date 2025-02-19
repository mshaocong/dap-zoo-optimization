module load Python3/3.8.15


PROJECT="zoo-mfem-dist"
BS=512 

# Function to run experiments for a given distribution
run_distribution() {
    local dist=$1
    for i in {1..10}; do
        echo "Running ${dist} iteration $i"
        python gd.py --fine_mesh_size 20 \
                     --coarse_mesh_size 10 \
                     --num_iterations 200 \
                     --batch_size $BS \
                     --learning_rate 0.1 \
                     --wandb_entity "scma" \
                     --wandb_project $PROJECT \
                     --wandb_tags "${dist}" \
                     --perturbation_distribution ${dist} &
        
        # Allow only 10 parallel jobs at a time (per distribution)
        if [[ $(jobs -r -p | wc -l) -ge 10 ]]; then
            wait -n
        fi
    done
    # Wait for all background jobs to complete
    wait
}

# Run all distributions in parallel
run_distribution "optimal" &
run_distribution "gaussian" &
run_distribution "uniform_sphere" &
run_distribution "rademacher" &
run_distribution "random_coordinate" &
 