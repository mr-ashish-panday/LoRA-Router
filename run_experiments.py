"""Run all experiments."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run LoRA Router experiments")
    parser.add_argument("--stage", choices=[
        "label", "train", "evaluate", "ablate", "analyze", "all"
    ], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for labeling")
    args = parser.parse_args()
    
    if args.stage in ["label", "all"]:
        print("\n" + "="*60)
        print("STAGE 1: Labeling Dataset")
        print("="*60)
        from src.data.labeler import label_dataset
        label_dataset(split="train", limit=args.limit)
    
    if args.stage in ["train", "all"]:
        print("\n" + "="*60)
        print("STAGE 2: Training Router")
        print("="*60)
        from src.train_router import train
        train()
    
    if args.stage in ["evaluate", "all"]:
        print("\n" + "="*60)
        print("STAGE 3: Evaluation")
        print("="*60)
        from src.eval.evaluate import run_full_evaluation
        run_full_evaluation()
    
    if args.stage in ["ablate", "all"]:
        print("\n" + "="*60)
        print("STAGE 4: Ablation Studies")
        print("="*60)
        from src.eval.ablations import run_all_ablations
        run_all_ablations()
    
    if args.stage in ["analyze", "all"]:
        print("\n" + "="*60)
        print("STAGE 5: Analysis")
        print("="*60)
        
        from src.analysis.interpretability import run_interpretability_analysis
        run_interpretability_analysis()
        
        from src.analysis.ood_eval import run_ood_evaluation
        run_ood_evaluation()
        
        from src.analysis.error_analysis import run_error_analysis
        run_error_analysis()
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
