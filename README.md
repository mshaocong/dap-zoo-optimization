# [ICLR 2025 Spotlight] Revisiting Zeroth-Order Optimization: Minimum-Variance Two-Point Estimators and Directionally Aligned Perturbations

This is the source code for our paper [*[ICLR 2025 Spotlight] Revisiting Zeroth-Order Optimization: Minimum-Variance Two-Point Estimators and Directionally Aligned Perturbations*](https://openreview.net/forum?id=ywFOSIT9ik).

> *Abstract:* In this paper, we explore the two-point zeroth-order gradient estimator and identify the distribution of random perturbations that minimizes the estimator's asymptotic variance as the perturbation stepsize tends to zero. We formulate it as a constrained functional optimization problem over the space of perturbation distributions. Our findings reveal that such desired perturbations can align directionally with the true gradient, instead of maintaining a fixed length. While existing research has largely focused on fixed-length perturbations, the potential advantages of directional alignment have been overlooked. To address this gap, we delve into the theoretical and empirical properties of the directionally aligned perturbation (DAP) scheme, which adaptively offers higher accuracy along critical directions. Additionally, we provide a convergence analysis for stochastic gradient descent using $\delta$-unbiased random perturbations, extending existing complexity bounds to a wider range of perturbations. Through empirical evaluations on both synthetic problems and practical tasks, we demonstrate that DAPs outperform traditional methods under specific conditions.
 
## Visualizing the DAP

To visualize the DAP, upload `DAP-Perturbations.ipynb` file to the Google Colab and run the cells. The notebook includes the parameter setting and will generate the DAP.

## Repeating the Synthetic Experiments

To repeat the synthetic experiments, upload `Synthetic-Experiments.ipynb` file to the Google Colab and run the cells. The notebook includes the customized function class `QuadraticFunction` and `ProductFunction`. All parameters are set in the notebook.

## Repeating the Language Model Training Experiment

We follow the [ZO-Bench](https://github.com/ZO-Bench/ZO-LLM) to implement the DAP-based zeroth-order optimization method. The full codes are included in the `llm-optimization-zoo` folder. To repeat our experiment, we use the following hyper-parameter settings:

```json
{
    "os": "Linux-4.18.0-553.5.1.el8_10.x86_64-x86_64-with-glibc2.28",
    "python": "3.10.10", 
    "args": [
        "--prompt_tuning",
        "--num_virtual_tokens=10",
        "--prompt_init_by_real_tokens",
        "--model_name=facebook/opt-1.3b",
        "--task_name=SST2", 
        "--overwrite_output_dir",
        "--no_reparam",
        "--num_train_epochs=5",
        "--per_device_train_batch_size=16",
        "--load_best_model_at_end",
        "--evaluation_strategy=steps",
        "--save_strategy=steps",
        "--save_total_limit=1",
        "--eval_steps=1000",
        "--max_steps=20000",
        "--logging_steps=10",
        "--num_eval=1000",
        "--num_train=1000",
        "--num_dev=500",
        "--train_as_classification",
        "--perturbation_mode=one_side",
        "--trainer=zo_sgd",
        "--optimizer=sgd",
        "--train_set_seed=1",
        "--lr_scheduler_type=constant",
        "--eval_steps=500",
        "--save_steps=500",
        "--learning_rate=1e-4",
        "--weight_decay=0",
        "--zo_eps=1e-5",
        "--perturbation=optimal"
    ], 
    "codePathLocal": "zo-bench/run.py",
    "codePath": "zo-bench/run.py"
}
```

For other perturbation methods, change the `--perturbation` parameter to `--perturbation=uniform`, `--perturbation=rademacher`, or `--perturbation=normal`.

Here, we provide the step-by-step instruction to repeat this experiment:

1. Install the required packages by running the following command:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the following command to train the model:

    ```bash
    srun python3 ./zo-bench/run.py --num_virtual_tokens=10 --prompt_init_by_real_tokens --model_name=facebook/opt-1.3b --task_name=SST2 --overwrite_output_dir --no_reparam --num_train_epochs=5 --per_device_train_batch_size=16 --load_best_model_at_end --evaluation_strategy=steps --save_strategy=steps --save_total_limit=1 --eval_steps=1000 --max_steps=20000 --logging_steps=10 --num_eval=1000 --num_train=1000 --num_dev=500 --train_as_classification --perturbation_mode=one_side --trainer=zo_sgd --optimizer=sgd --train_set_seed=0 --lr_scheduler_type=constant --eval_steps=500 --save_steps=500 --learning_rate=1e-4 --weight_decay=0 --zo_eps=1e-5 --perturbation=rademacher
    ```

## Repeating the Mesh Optimization Experiment
After installing required packages, run
```bash
sbatch run.sh
```

## Citation
If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{
ma2025revisiting,
    title={Revisiting Zeroth-Order Optimization:  Minimum-Variance Two-Point Estimators and  Directionally Aligned Perturbations},
    author={Shaocong Ma and Heng Huang},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=ywFOSIT9ik}
}
```

## Licensing Notice

This repository contains two parts:

* Our Codes: Licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
* Third-Party Code ([ZO-Bench](https://github.com/ZO-Bench/ZO-LLM)): Located in the `llm-optimization-zoo` folder, licensed under GPLv3. Please refer to the LICENSE file inside that folder for details.
