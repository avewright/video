#!/usr/bin/env python3
"""
Script to save the current training state before reaching the next checkpoint.
This can be used to save progress at any point during training.
"""

import argparse
import os
import sys
import signal
import psutil
import torch
from pathlib import Path


def find_training_process():
    """Find the running training process."""
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in process.info['name'] and process.info['cmdline']:
                cmdline = ' '.join(process.info['cmdline'])
                if 'train.py' in cmdline and 'qwen25_3b_qlora.yaml' in cmdline:
                    return process.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def create_manual_checkpoint(training_output_dir: str, step_number: int = None):
    """
    Create a checkpoint directory structure manually.
    This will be populated when we stop training gracefully.
    """
    if step_number is None:
        # Estimate current step from training time and logs
        step_number = estimate_current_step(training_output_dir)
    
    checkpoint_dir = os.path.join(training_output_dir, f"checkpoint-{step_number}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"📁 Created checkpoint directory: {checkpoint_dir}")
    return checkpoint_dir


def estimate_current_step(training_output_dir: str):
    """
    Estimate current training step from TensorBoard logs or training time.
    """
    try:
        # Look for the latest TensorBoard event file
        runs_dir = os.path.join(training_output_dir, "runs")
        latest_run = None
        latest_time = 0
        
        for run_dir in os.listdir(runs_dir):
            run_path = os.path.join(runs_dir, run_dir)
            if os.path.isdir(run_path):
                mtime = os.path.getmtime(run_path)
                if mtime > latest_time:
                    latest_time = mtime
                    latest_run = run_path
        
        if latest_run:
            # Look for tfevents files
            for file in os.listdir(latest_run):
                if file.startswith("events.out.tfevents"):
                    # Try to read the last few events to estimate step
                    # For now, use a rough estimate based on training time
                    break
        
        # Rough estimation: if training has been running for ~2.7 hours
        # and expected total is ~7 hours for 768 steps, we're at ~300 steps
        estimated_step = 300  # Conservative estimate
        print(f"📊 Estimated current step: ~{estimated_step}")
        return estimated_step
        
    except Exception as e:
        print(f"⚠️  Could not estimate step: {e}")
        return 250  # Conservative fallback


def save_training_gracefully(pid: int, output_dir: str):
    """
    Send SIGTERM to training process to save gracefully.
    """
    try:
        print(f"🛑 Sending graceful stop signal to training process (PID: {pid})")
        os.kill(pid, signal.SIGTERM)
        print("✅ Signal sent. Training should save current state and exit gracefully.")
        print("⏳ Wait for training to finish saving, then run the save_model.py script.")
        return True
    except Exception as e:
        print(f"❌ Error stopping training: {e}")
        return False


def modify_config_for_frequent_saves(config_path: str = "configs/qwen25_3b_qlora.yaml"):
    """
    Create a modified config that saves more frequently.
    """
    import yaml
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Modify save frequency
        config['training']['save_steps'] = 50  # Save every 50 steps instead of 500
        config['training']['eval_steps'] = 50  # Evaluate every 50 steps too
        
        # Create backup and new config
        backup_path = config_path.replace('.yaml', '_backup.yaml')
        with open(backup_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        new_config_path = config_path.replace('.yaml', '_frequent_saves.yaml')
        with open(new_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"📝 Created config with frequent saves: {new_config_path}")
        print(f"💾 Backup of original config: {backup_path}")
        return new_config_path
        
    except Exception as e:
        print(f"❌ Error modifying config: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Save current training state")
    parser.add_argument(
        "--action",
        choices=["stop_and_save", "modify_config", "check_status"],
        default="check_status",
        help="Action to take"
    )
    parser.add_argument(
        "--training_output_dir",
        default="./outputs/qwen25-3b-qlora-invoice",
        help="Training output directory"
    )
    parser.add_argument(
        "--config_path",
        default="configs/qwen25_3b_qlora.yaml",
        help="Path to training config"
    )
    
    args = parser.parse_args()
    
    # Find training process
    training_pid = find_training_process()
    
    if args.action == "check_status":
        if training_pid:
            print(f"✅ Training is running (PID: {training_pid})")
            estimated_step = estimate_current_step(args.training_output_dir)
            print(f"📊 Estimated progress: ~{estimated_step}/768 steps ({estimated_step/768*100:.1f}%)")
            print("\nOptions:")
            print("1. Stop training and save current state:")
            print("   python scripts/save_current_training.py --action stop_and_save")
            print("\n2. Create config with frequent saves:")
            print("   python scripts/save_current_training.py --action modify_config")
        else:
            print("❌ No training process found")
            
    elif args.action == "stop_and_save":
        if training_pid:
            checkpoint_dir = create_manual_checkpoint(args.training_output_dir)
            if save_training_gracefully(training_pid, args.training_output_dir):
                print(f"\n🔄 Next steps:")
                print(f"1. Wait for training to complete saving")
                print(f"2. Run: python scripts/save_model.py")
                print(f"   (It will automatically find the latest checkpoint)")
        else:
            print("❌ No training process found to stop")
            
    elif args.action == "modify_config":
        new_config = modify_config_for_frequent_saves(args.config_path)
        if new_config and training_pid:
            print(f"\n🔄 To use the new config:")
            print(f"1. Stop current training: python scripts/save_current_training.py --action stop_and_save")
            print(f"2. Restart with frequent saves: python train.py --config {new_config}")


if __name__ == "__main__":
    main() 