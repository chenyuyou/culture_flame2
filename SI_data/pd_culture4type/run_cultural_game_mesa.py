import os
import pandas as pd
import numpy as np
import argparse # 导入 argparse 模块
from cultural_game_mesa import CulturalGameModel # 确保导入你的模型类

def run_simulation(args):
    """
    Runs a single simulation with the given parameters and saves data.
    """
    print(f"\n--- Running simulation for b = {args.b:.2f} ---")

    # 将命令行参数转换为模型所需的字典
    model_params = {
        "L": args.L,
        "initial_coop_ratio": args.initial_coop_ratio,
        "b": args.b,
        "K": args.K,
        "K_C": args.K_C,
        "p_update_C": args.p_update_C,
        "p_mut_culture": args.p_mut_culture,
        "p_mut_strategy": args.p_mut_strategy,
        "C_dist": args.C_dist,
        "mu": args.mu,
        "sigma": args.sigma,
        "seed": args.seed,
    }

    # 初始化模型
    model = CulturalGameModel(**model_params)

    # 运行模拟
    for i in range(args.num_steps):
        model.step()
        # 如果需要保存中间时间步的快照，可以在这里添加条件
        # 例如：if i % 100 == 0: save_snapshot(model, i, args.b)

    # --- 设置输出目录 ---
    # 目录名可以包含一些关键参数，以便区分不同的运行
    output_subdir_name = f"L{args.L}_b{args.b:.2f}_K{args.K}_KC{args.K_C}_steps{args.num_steps}"
    output_base_dir = os.path.join("simulation_results", output_subdir_name)
    spatial_snapshot_dir = os.path.join(output_base_dir, "spatial_snapshots_data")
    model_data_dir = os.path.join(output_base_dir, "model_data")

    os.makedirs(spatial_snapshot_dir, exist_ok=True)
    os.makedirs(model_data_dir, exist_ok=True)

    # --- 保存数据 ---
    # 1. 保存最终状态的空间快照数据
    final_grid_state = model.get_grid_state_for_snapshot()
    snapshot_filename = os.path.join(spatial_snapshot_dir, f"snapshot_b_{args.b:.2f}_step_{args.num_steps}.npy")
    np.save(snapshot_filename, final_grid_state)
    print(f"Saved final spatial snapshot for b = {args.b:.2f} to {snapshot_filename}")

    # 2. 保存模型级别的时间序列数据
    model_df = model.datacollector.get_model_vars_dataframe()
    model_data_filename = os.path.join(model_data_dir, f"model_data_b_{args.b:.2f}.csv")
    model_df.to_csv(model_data_filename, index_label="Step")
    print(f"Saved model data for b = {args.b:.2f} to {model_data_filename}")

    print(f"Simulation for b = {args.b:.2f} completed.")
    print(f"Results saved to: {output_base_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a cultural game Mesa simulation.")

    # --- 定义命令行参数 ---
    # 核心参数
    parser.add_argument("--L", type=int, default=50,
                        help="Grid size (L x L).")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Total number of simulation steps.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    # 模型参数
    parser.add_argument("--initial_coop_ratio", type=float, default=0.5,
                        help="Initial ratio of cooperators.")
    parser.add_argument("--b", type=float, default=2.0, # 默认值可以根据需要设置
                        help="Benefit for cooperation (b parameter).")
    parser.add_argument("--K", type=float, default=0.1,
                        help="Selection strength for strategy update.")
    parser.add_argument("--K_C", type=float, default=0.1,
                        help="Selection strength for cultural update.")
    parser.add_argument("--p_update_C", type=float, default=0.5,
                        help="Probability of attempting cultural update.")
    parser.add_argument("--p_mut_culture", type=float, default=0.005,
                        help="Probability of cultural mutation.")
    parser.add_argument("--p_mut_strategy", type=float, default=0.001,
                        help="Probability of strategy mutation.")
    parser.add_argument("--C_dist", type=str, default="uniform",
                        choices=["uniform", "bimodal", "normal", "fixed"],
                        help="Cultural value distribution type.")
    parser.add_argument("--mu", type=float, default=0.5,
                        help="Mean for normal/bimodal/fixed C_dist.")
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="Standard deviation for normal C_dist.")

    # 批量运行参数 (可选，用于一次运行多个 b 值)
    parser.add_argument("--b_values", type=float, nargs='*',
                        help="List of b values to run simulations for. If provided, --b will be ignored.")

    args = parser.parse_args()

    print("Starting simulation(s) with the following parameters:")
    for arg_name, arg_value in vars(args).items():
        print(f"  {arg_name}: {arg_value}")

    if args.b_values:
        # 如果提供了 b_values 列表，则遍历运行
        print("\nRunning multiple simulations for specified b_values...")
        original_b = args.b # 保存原始的 b 值，以便在循环中恢复
        for b_val in args.b_values:
            args.b = b_val # 更新 args.b 以便 run_simulation 使用
            run_simulation(args)
        args.b = original_b # 恢复 b 值，以防后续代码需要
    else:
        # 否则，只运行一次，使用 --b 参数指定的值
        run_simulation(args)

    print("\nAll requested simulations completed and data saved.")
    print("To generate the spatial snapshot plots, run 'plot_spatial_snapshots.py' (you'll need to specify the paths to the .npy files).")

