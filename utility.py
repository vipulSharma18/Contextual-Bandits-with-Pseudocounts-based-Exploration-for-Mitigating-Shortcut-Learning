import pandas as pd
import numpy as np
import wandb

api = wandb.Api()

run_id = "k4621ghy"
run = api.run(f"vipul/RL_Project_CSCI2951F/{run_id}")
history_df = run.history()
history_df.to_csv("measure_bias/bias_data_threshold.csv", index=False)