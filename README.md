# ICORES_zipper

各サブプロットで理論とシミュレーションを30点、各シミュレーションは1,100,000イベント（ウォームアップ100,000）

PYTHONPATH=. /opt/conda/bin/python - << 'PY'
from plot_simulation import plot_6_theory_vs_sim
plot_6_theory_vs_sim(sim_max_events=1100000, sim_warmup=100000, num_sim_points=30, seed_base=123)
PY
