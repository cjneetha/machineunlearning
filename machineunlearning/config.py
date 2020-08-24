import os

run_path = "/Users/amused_confused/Documents/OVGU/Hiwi/oscar/runs/run1"
datafile_path = "/Users/amused_confused/Documents/OVGU/Hiwi/oscar/data/test"

log_file = os.path.join(run_path, "run1.log")
predictions_csv_file = os.path.join(run_path, "run1.csv")

contamination_files = ['20121216.pkl.gzip']

topk = 12000
entity_min_instances = 40

to_class = 'negative'
portion = 0.9

