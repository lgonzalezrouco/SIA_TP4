import argparse
import sys

from hopfield.experiment import HopfieldExperiment
from kohonen.experiment import KohonenExperiment
from oja.experiment import OjaExperiment
from oja.pca_analysis import PCAAnalyzer
from utils.config_parser import ConfigParser
from utils.data_loader import load_europe_data, preprocess_data


def main():
    parser = argparse.ArgumentParser(description="SIA TP4 - Unsupervised Learning")
    parser.add_argument("--config", type=str, default="config.example.json", help="Path to config file")
    parser.add_argument("--exercise", type=str, choices=["kohonen", "oja", "hopfield", "pca"], help="Exercise to run")
    
    args = parser.parse_args()
    
    try:
        config_obj = ConfigParser(args.config)
        config = config_obj.get_global_config()
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
        
    exercise = args.exercise or config.get("exercise", "kohonen")
    
    print(f"Starting SIA TP4 - Exercise: {exercise}")
    
    if exercise == "pca":
        df = load_europe_data()
        if df is not None:
            analyzer = PCAAnalyzer(df)
            analyzer.run()
            
    elif exercise in ["kohonen", "oja"]:
        exercise_config = config_obj.get_exercise_config(exercise)
        df = load_europe_data()
        if df is not None:
            data, labels = preprocess_data(df)
            if exercise == "kohonen":
                exp = KohonenExperiment(exercise_config, data, labels)
            else:
                exp = OjaExperiment(exercise_config, df)
            exp.run()
            
    elif exercise == "hopfield":
        exercise_config = config_obj.get_exercise_config(exercise)
        exp = HopfieldExperiment(exercise_config)
        exp.run()
    else:
        print(f"Unknown exercise: {exercise}")

if __name__ == "__main__":
    main()
