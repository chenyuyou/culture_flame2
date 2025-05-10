# culture_flame2

# 图1、2、7共6个图
## 运行数据
python cultural_game_ensemble.py --output_dir my_simulation_logs1 --steps 500 --num_runs 1000

## 数据转换
python process_logs.py --log_dir my_simulation_logs1 --output_file processed_data.pkl --steady_state_window 200

## 绘图数据
python plot_specific_figures.py --data_file my_simulation_logs1/processed_data.pkl --output_dir my_specific_figures1 --plot all