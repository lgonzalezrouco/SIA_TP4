import argparse
import json
import os
import sys

from utils.data_loader import load_europe_data, preprocess_data
from kohonen.som import run_kohonen
from oja.oja_neuron import run_oja
from hopfield.hopfield_net import run_hopfield

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="SIA TP4 - Unsupervised Learning")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--exercise", type=str, choices=["kohonen", "oja", "hopfield"], help="Exercise to run")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        sys.exit(1)
        
    config = load_config(args.config)
    exercise = args.exercise or config.get("exercise", "kohonen")
    
    print(f"Starting SIA TP4 - Exercise: {exercise}")
    
    if exercise in ["kohonen", "oja"]:
        df = load_europe_data()
        if df is not None:
            normalized_data, labels = preprocess_data(df)
            if exercise == "kohonen":
                run_kohonen(config.get("kohonen", {}), normalized_data)
            else:
                run_oja(config.get("oja", {}), normalized_data)
    elif exercise == "hopfield":
        run_hopfield(config.get("hopfield", {}))
    else:
        print(f"Unknown exercise: {exercise}")

if __name__ == "__main__":
    main()
