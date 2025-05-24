
# snapshot图
## 运行并收集数据
python cultural_game_simulation_snapshot.py --b_range 1.1 8 0.1 --steps 500

python cultural_game_simulation_snapshot.py --b_values 1.5,1.9,7.0 --step 500
## 绘图
python plot_snap_b_values.py --b_values 1.9 2.5 7.0 --step 500

