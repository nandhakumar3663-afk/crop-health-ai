"""
backend/model_class_mapping.py - Semantic Class Cross-Mapping
Translates predictions between PlantDoc-ResNet50 (29 classes) and FarmGuardian (38 classes).
Handles semantic matching without raw index comparison and flags unmappable classes.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

MAPPING_FILE = Path(__file__).resolve().parent / "plantdoc_class_mapping.json"


class ModelClassMapper:
    def __init__(self, mapping_path: Optional[Path] = None):
        self.mapping_path = mapping_path or MAPPING_FILE
        self.plantdoc_to_fg: Dict[str, Dict[str, Any]] = {}
        self.unmappable_fg: set = set()
        self._load()

    def _load(self):
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found at: {self.mapping_path}")
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.plantdoc_to_fg = data.get("plantdoc_to_farmguardian", {})
        self.unmappable_fg = set(data.get("unmappable_farmguardian_classes", []))

    def get_plantdoc_class_info(self, plantdoc_idx: int) -> Dict[str, Any]:
        """Return full mapping metadata for a PlantDoc class index."""
        str_idx = str(plantdoc_idx)
        if str_idx in self.plantdoc_to_fg:
            return self.plantdoc_to_fg[str_idx]
        return {
            "plantdoc_class": f"Unknown_{plantdoc_idx}",
            "farmguardian_class": "not_mappable",
            "mappable": False
        }

    def map_plantdoc_to_farmguardian(self, plantdoc_idx: int) -> Optional[str]:
        """Translate a PlantDoc class index into FarmGuardian raw class name."""
        info = self.get_plantdoc_class_info(plantdoc_idx)
        if info.get("mappable", False):
            return info.get("farmguardian_class")
        return "not_mappable"

    def compare_predictions(
        self,
        primary_fg_raw: str,
        plantdoc_idx: int
    ) -> Tuple[bool, str, str]:
        """
        Compare primary prediction against secondary PlantDoc prediction.
        Returns:
            (agree: bool, mapped_fg_class: str, plantdoc_display_name: str)
        """
        info = self.get_plantdoc_class_info(plantdoc_idx)
        plantdoc_display = info.get("plantdoc_class", f"Class_{plantdoc_idx}")
        mapped_fg = info.get("farmguardian_class", "not_mappable")

        if mapped_fg == "not_mappable":
            return False, "not_mappable", plantdoc_display

        # Check semantic agreement
        agree = (primary_fg_raw.strip().lower() == mapped_fg.strip().lower())
        return agree, mapped_fg, plantdoc_display


_mapper_instance = None


def get_class_mapper() -> ModelClassMapper:
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = ModelClassMapper()
    return _mapper_instance
