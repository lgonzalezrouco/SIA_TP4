import json
import os


class ConfigParser:
    # TODO: Define a specific type (e.g., TypedDict or DataClass) for the configuration
    # to improve autocompletion and type safety throughout the project.

    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_and_validate()

    def _load_and_validate(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file {self.config_path} not found.")

        with open(self.config_path, "r") as f:
            config = json.load(f)

        # Validaciones básicas de estructura
        if "exercise" not in config:
            config["exercise"] = "kohonen"

        # Asegurar que existan las secciones para cada ejercicio
        for ex in ["kohonen", "oja", "hopfield"]:
            if ex not in config:
                config[ex] = {}

        return config

    def get_exercise_config(self, exercise):
        return self.config.get(exercise, {})

    def get_global_config(self):
        return self.config
