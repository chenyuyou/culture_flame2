**如何使用这些文件：**

1.  将上述代码分别保存到 `run_main_scan.py` 和 `run_phasediagram_scan.py` 文件中。
2.  确保您已经有了 `model_definition.py`, `host_functions.py`, `parameter_sweep.py`, 和 `cultural_game_cuda_array.py` 文件。
3.  打开终端或命令提示符。
4.  导航到包含这些文件的目录。
5.  要运行主扫描（非相图数据），执行：
    ```bash
    python run_main_scan.py
    ```
    您可以使用命令行参数覆盖默认设置。
6.  要运行相图扫描，执行：
    ```bash
    python run_phasediagram_scan.py
    ```
    您可以使用命令行参数覆盖默认设置。
7.  模拟运行完成后，原始日志将保存在各自指定的输出目录下的子目录中。



8.  根据您想要处理的数据类型，运行 `process_logs.py`：

    *   **处理主扫描的稳态数据：**
        ```bash
        python process_logs.py --log_dir results_flamegpu_main_scan_raw --process_type steady_state --output_file_steady_state main_scan_steady_state.pkl
        ```
    *   **处理主扫描的时间序列数据：**
        ```bash
        python process_logs.py --log_dir results_flamegpu_main_scan_raw --process_type time_series --output_file_timeseries main_scan_timeseries.pkl
        ```
    *   **处理相图扫描的稳态数据：**
        ```bash
        python process_logs.py --log_dir results_flamegpu_phasediagram_ensemble_raw --process_type steady_state --output_file_steady_state phasediagram_steady_state.pkl
        ```
    *   **处理某个目录下的所有数据（稳态和时间序列）：**
        ```bash
        python process_logs.py --log_dir results_flamegpu_main_scan_raw --process_type all --output_file_steady_state main_scan_steady_state.pkl --output_file_timeseries main_scan_timeseries.pkl
        ```
---

## 相图
python run_phasediagram_scan.py --output_dir test_phase1 --step 200 --num_runs 200  

python process_logs.py --log_dir test_phase1 --process_type steady_state        

python plot_specific_figures.py --data_file test_phase1/processed_steady_state_results.pkl --output_dir specific_figures1 --plot all

## 其他稳态图



