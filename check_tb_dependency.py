
from accelerate import Accelerator
try:
    accelerator = Accelerator(log_with="tensorboard")
    accelerator.init_trackers("test_project")
    print("Success: Tracker initialized")
except Exception as e:
    print(f"Error: {e}")
