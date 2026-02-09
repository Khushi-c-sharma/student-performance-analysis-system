import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Union, Dict, List
from .config import PROJECT_ROOT


class AnalysisPersistenceManager:
    """
    Hardened manager for analytical outputs.
    Features: Atomic writes, path sanitization, and timestamp-based versioning.
    """

    def __init__(self, base_output_dir: str = "outputs", versioned: bool = True):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.versioned = versioned

        try:
            # 1. Define the path
            self.base_dir = (PROJECT_ROOT / Path(base_output_dir)).resolve()

            # 2. Define sub - directories
            self.csv_dir = self.base_dir / "analysis" / "tables"
            self.json_dir = self.base_dir / "analysis" / "summaries"

            # 3. ACTUALLY create them on the disk
            self._create_directories()
        except Exception as e:
            self.logger.critical("Initialization failed: %s", e)
            raise

    def _create_directories(self) -> None:
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)

    def _get_timestamped_name(self, name: str) -> str:
        """Appends a sortable ISO-like timestamp if versioning is enabled."""
        if not self.versioned:
            return name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{name}_{timestamp}"

    def persist(self, artifact: Any, name: str) -> None:
        """Routes and persists artifacts with safety checks."""
        # Sanitize name to prevent directory traversal
        clean_name = Path(name).name
        final_name = self._get_timestamped_name(clean_name)

        if isinstance(artifact, pd.DataFrame):
            self._save_csv(artifact, final_name)
        elif isinstance(artifact, (dict, list)):
            self._save_json(artifact, final_name)
        else:
            self.logger.error("Unsupported type %s for: %s", type(artifact), clean_name)
            raise TypeError(f"Persistence not implemented for {type(artifact)}")

    def _save_csv(self, df: pd.DataFrame, name: str) -> None:
        path = self.csv_dir / f"{name}.csv"
        temp_path = path.with_suffix(".tmp")

        try:
            df.to_csv(temp_path, index=False)
            temp_path.replace(path)
            self.logger.info("Persisted CSV: %s", path.name)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            self.logger.error("CSV write failed for %s: %s", name, e)
            raise

    def _save_json(self, data: Union[Dict, List], name: str) -> None:
        path = self.json_dir / f"{name}.json"
        temp_path = path.with_suffix(".tmp")

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, default=str)

            temp_path.replace(path)
            self.logger.info("Persisted JSON: %s", path.name)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            self.logger.error("JSON write failed for %s: %s", name, e)
            raise
