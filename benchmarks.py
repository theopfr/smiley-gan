
import json
import hashlib
import time
import os


class Benchmark:
    def __init__(self, benchmark_file: str="", clear_all: bool=False):
        self.benchmark_file = benchmark_file

        if clear_all:
            self._delete_benchmarks(save_time=5)

        if not os.path.exists(self.benchmark_file):
            self._create_benchmark_file()

    # remove all benchmarks
    def _delete_benchmarks(self, save_time: int=3):
        print("Deleting all benchmarks in " + str(save_time) + " seconds!")
        time.sleep(save_time)

        os.system("rm " + self.benchmark_file)
        os.system("rm -r " + self.benchmark_file.split("/")[0] + "/generated_images/*")
        os.system("rm -r " + self.benchmark_file.split("/")[0] + "/models/*")
        os.system("rm " + self.benchmark_file.split("/")[0] + "/plots/*")

    # create benchmark json file if it does not exist
    def _create_benchmark_file(self):
        with open(self.benchmark_file, "w+") as f:
            init = {}
            json.dump(init, f, indent=4)

    # load content from benchmark file
    def _load_json_content(self) -> dict:
        with open(self.benchmark_file, "r") as f:
            return json.load(f)

    # saving entries to benchmark file
    def _save_json_content(self, content: dict):
        with open(self.benchmark_file, "w") as f:
            json.dump(content, f, indent=4)

    # create entry id
    def create_id(self) -> str:
        entries = self._load_json_content()
        entry_index = len(entries) + 1

        return str(entry_index)

    # create entry
    def create_entry(self, id_: str, optimizer, loss_function, epochs: int, batch_size: int, disc_lr: float, gen_lr: float, disc_lr_decay: float, gen_lr_decay: float, lr_decay_period: int, gaussian_noise_range: tuple):
        entries = self._load_json_content()
        entry = {
                    "hyperparameters": {
                        "optimizer": str(optimizer).split("(")[0],
                        "loss-function": str(loss_function).split("(")[0],
                        "epochs": epochs,
                        "batch-size": batch_size,
                        "disc-lr": disc_lr,
                        "gen-lr": gen_lr,
                        "disc-lr-decay": disc_lr_decay,
                        "gen-lr-decay": gen_lr_decay,
                        "lr-decay-period": lr_decay_period,
                        "linear-gaussian-noise-range": [gaussian_noise_range[0], gaussian_noise_range[1]]
                    },
                    "generated-images-folder": ("generated_images/" + id_),
                    "plots": ("plots/" + id_ + ".png"),
                    "models": "models/" + id_
                }

        entries[id_] = entry

        self._save_json_content(entries)
        