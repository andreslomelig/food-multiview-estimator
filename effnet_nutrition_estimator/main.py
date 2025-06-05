# main.py
import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiview Nutrition Estimator")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], required=True,
                        help="Choose whether to train or evaluate the model.")
    args = parser.parse_args()

    if args.mode == "train":
        subprocess.run(["python", "training/train_multiview_effnet.py"])
    elif args.mode == "eval":
        subprocess.run(["python", "evaluation/eval_multiview_effnet.py"])
