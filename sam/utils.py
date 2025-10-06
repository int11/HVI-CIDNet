import os
import sys
import torch
from datetime import datetime
import dist


class Tee:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, 'a')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self  # Redirect stdout to this instance
        print(f"===== Logging session started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        return self

    def write(self, obj):
        self.file.write(obj)
        self.file.flush()
        self.stdout.write(obj)
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"===== Logging session ended {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        sys.stdout = self.stdout  # Restore original stdout
        self.file.close()


def checkpoint(epoch, model, optimizer, path):
    os.makedirs(path, exist_ok=True)
    model_out_path = os.path.join(path, f"epoch_{epoch}.pth")

    # Save model and optimizer states with epoch info
    # Use de_parallel to handle distributed models
    model_state = dist.de_parallel(model).state_dict() if dist.is_dist_available_and_initialized() else model.state_dict()

    checkpoint_dict = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint_dict, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
