"""
Vision module — Symbolic Grid Parser and Object Correspondence.

TASK 5: Added find_object_correspondence() function that matches
objects between input/output grids and identifies transformation types.
"""

import numpy as np
from scipy.ndimage import label, find_objects
from typing import List, Dict, Any, Tuple

class GridObject:
    """Represents a discrete contiguous shape on the ARC grid."""
    def __init__(self, obj_id: int, color: int, mask: np.ndarray, bounding_box: tuple):
        self.obj_id = obj_id
        self.color = color
        self.mask = mask
        self.bounding_box = bounding_box
        
        # Calculate properties
        self.size = int(np.sum(self.mask))
        r_slice, c_slice = self.bounding_box
        self.top, self.bottom = r_slice.start, r_slice.stop - 1
        self.left, self.right = c_slice.start, c_slice.stop - 1
        self.width = self.right - self.left + 1
        self.height = self.bottom - self.top + 1
        self.center_y = self.top + (self.height / 2)
        self.center_x = self.left + (self.width / 2)
        
        # Advanced Geometry
        self.is_square = (self.width == self.height) and (self.size == self.width * self.height)
        self.is_rectangle = (self.size == self.width * self.height)
        self.is_horizontal_line = (self.height == 1 and self.width > 1)
        self.is_vertical_line = (self.width == 1 and self.height > 1)
        
        # Basic Symmetry Check
        cropped_mask = self.mask[self.top:self.bottom+1, self.left:self.right+1]
        self.is_symmetric_h = np.array_equal(cropped_mask, np.fliplr(cropped_mask))
        self.is_symmetric_v = np.array_equal(cropped_mask, np.flipud(cropped_mask))

    def to_dict(self) -> Dict[str, Any]:
        """Returns a symbolic representation easy for an LLM to read."""
        return {
            "id": self.obj_id,
            "color": self.color,
            "size": self.size,
            "position": {"top": self.top, "left": self.left},
            "dimensions": {"width": self.width, "height": self.height},
            "center_of_mass": {"y": self.center_y, "x": self.center_x},
            "geometry": {
                "is_square": self.is_square,
                "is_rectangle": self.is_rectangle,
                "is_horizontal_line": self.is_horizontal_line,
                "is_vertical_line": self.is_vertical_line,
                "is_symmetric_h": self.is_symmetric_h,
                "is_symmetric_v": self.is_symmetric_v
            }
        }


class SymbolicGridParser:
    """
    Parses a 2D integer numpy array representing an ARC grid into
    an Object-Oriented Spatial Graph.
    """
    def __init__(self, ignore_background_color: int = 0):
        self.bg_color = ignore_background_color

    def parse(self, grid: List[List[int]]) -> Dict[str, Any]:
        """Takes raw 2D grid, outputs a heavily structured symbolic graph."""
        np_grid = np.array(grid, dtype=np.int32)
        objects = self._extract_objects(np_grid)
        relationships = self._compute_spatial_relationships(objects)
        
        return {
            "grid_dimensions": {"height": np_grid.shape[0], "width": np_grid.shape[1]},
            "background_color": self.bg_color,
            "objects": [obj.to_dict() for obj in objects],
            "relationships": relationships
        }

    def _extract_objects(self, grid: np.ndarray) -> List[GridObject]:
        """Finds all contiguous blocks of the same color."""
        detected_objects = []
        obj_counter = 1
        
        for color in range(10):
            if color == self.bg_color:
                continue
                
            color_mask = (grid == color).astype(int)
            if np.sum(color_mask) == 0:
                continue
                
            structure = np.ones((3, 3), dtype=int) 
            labeled_array, num_features = label(color_mask, structure=structure)
            slices = find_objects(labeled_array)
            
            for i, slc in enumerate(slices):
                if slc is not None:
                    obj_mask = (labeled_array == (i + 1))
                    obj = GridObject(
                        obj_id=obj_counter,
                        color=color,
                        mask=obj_mask,
                        bounding_box=slc
                    )
                    detected_objects.append(obj)
                    obj_counter += 1
                    
        return detected_objects

    def _compute_spatial_relationships(self, objects: List[GridObject]) -> List[Dict[str, Any]]:
        """Derives high-level spatial rules from the extracted objects."""
        relations = []
        
        for i, obj_a in enumerate(objects):
            for j, obj_b in enumerate(objects):
                if i == j:
                    continue
                    
                if obj_a.bottom < obj_b.top:
                    relations.append({"subject_id": obj_a.obj_id, "relation": "is_above", "target_id": obj_b.obj_id})
                if obj_a.top > obj_b.bottom:
                    relations.append({"subject_id": obj_a.obj_id, "relation": "is_below", "target_id": obj_b.obj_id})
                if obj_a.right < obj_b.left:
                    relations.append({"subject_id": obj_a.obj_id, "relation": "is_left_of", "target_id": obj_b.obj_id})
                if obj_a.left > obj_b.right:
                    relations.append({"subject_id": obj_a.obj_id, "relation": "is_right_of", "target_id": obj_b.obj_id})
                if (obj_a.top >= obj_b.top and obj_a.bottom <= obj_b.bottom and
                    obj_a.left >= obj_b.left and obj_a.right <= obj_b.right):
                    relations.append({"subject_id": obj_a.obj_id, "relation": "is_inside_bounds_of", "target_id": obj_b.obj_id})

        return relations

    def parse_pairs(self, training_pairs: List[Tuple[Any, Any]]) -> Dict[str, Any]:
        """
        Accepts a list of (input_grid, output_grid) training pairs.
        Parses all pairs and returns a combined symbolic graph showing what changed.
        """
        combined_graph = {"pairs": []}
        
        for i, (in_grid, out_grid) in enumerate(training_pairs):
            in_parsed = self.parse(in_grid)
            out_parsed = self.parse(out_grid)
            
            in_obj_count = len(in_parsed['objects'])
            out_obj_count = len(out_parsed['objects'])
            
            changes = {
                "object_count_delta": out_obj_count - in_obj_count,
                "input_dimensions": in_parsed["grid_dimensions"],
                "output_dimensions": out_parsed["grid_dimensions"],
                "dimension_changed": in_parsed["grid_dimensions"] != out_parsed["grid_dimensions"]
            }
            
            combined_graph["pairs"].append({
                "pair_index": i,
                "input_graph": in_parsed,
                "output_graph": out_parsed,
                "observed_changes": changes
            })
            
        return combined_graph


# ═══════════════════════════════════════════════════════
# TASK 5 — OBJECT CORRESPONDENCE
# ═══════════════════════════════════════════════════════

def _extract_objects_for_correspondence(grid: List[List[int]], bg_color: int = 0) -> List[Dict[str, Any]]:
    """
    Extract connected components from a grid as object descriptors.
    Returns a list of dicts with: color, bounding_box, centroid, size, mask_shape.
    """
    np_grid = np.array(grid, dtype=np.int32)
    objects = []
    obj_counter = 0

    for color in range(10):
        if color == bg_color:
            continue
        
        color_mask = (np_grid == color).astype(int)
        if np.sum(color_mask) == 0:
            continue
        
        structure = np.ones((3, 3), dtype=int)
        labeled_array, num_features = label(color_mask, structure=structure)
        slices = find_objects(labeled_array)
        
        for i, slc in enumerate(slices):
            if slc is None:
                continue
            
            obj_mask = (labeled_array == (i + 1))
            size = int(np.sum(obj_mask))
            
            r_slice, c_slice = slc
            top, bottom = r_slice.start, r_slice.stop - 1
            left, right = c_slice.start, c_slice.stop - 1
            width = right - left + 1
            height = bottom - top + 1
            center_y = top + height / 2.0
            center_x = left + width / 2.0
            
            # Extract the cropped mask shape for matching
            cropped = obj_mask[top:bottom+1, left:right+1]
            
            objects.append({
                "id": obj_counter,
                "color": color,
                "size": size,
                "bounding_box": {"top": top, "bottom": bottom, "left": left, "right": right},
                "dimensions": {"width": width, "height": height},
                "centroid": {"y": center_y, "x": center_x},
                "mask_shape": cropped.tolist()
            })
            obj_counter += 1
    
    return objects


def find_object_correspondence(input_grid: List[List[int]], 
                                output_grid: List[List[int]],
                                bg_color: int = 0) -> List[Dict[str, Any]]:
    """
    TASK 5: Find object correspondences between input and output grids.
    
    Steps:
      1. Extract connected components from both grids
      2. For each input object: color, bounding box, centroid, size
      3. For each output object: same features
      4. Match by scoring: color_match*3 + shape_match*2 + proximity*1
      5. Return list of dicts with input_obj, output_obj, confidence, transform_type
    """
    input_objects = _extract_objects_for_correspondence(input_grid, bg_color)
    output_objects = _extract_objects_for_correspondence(output_grid, bg_color)
    
    correspondences = []
    matched_output_ids = set()
    
    # For each input object, find best matching output object
    for in_obj in input_objects:
        best_match = None
        best_score = -1
        
        for out_obj in output_objects:
            if out_obj["id"] in matched_output_ids:
                continue
            
            score = 0.0
            
            # Color match (weight 3)
            if in_obj["color"] == out_obj["color"]:
                score += 3.0
            
            # Shape match (weight 2) — compare dimensions
            w_match = 1.0 - min(abs(in_obj["dimensions"]["width"] - out_obj["dimensions"]["width"]) / 
                               max(in_obj["dimensions"]["width"], 1), 1.0)
            h_match = 1.0 - min(abs(in_obj["dimensions"]["height"] - out_obj["dimensions"]["height"]) / 
                               max(in_obj["dimensions"]["height"], 1), 1.0)
            size_match = 1.0 - min(abs(in_obj["size"] - out_obj["size"]) / 
                                   max(in_obj["size"], 1), 1.0)
            shape_score = (w_match + h_match + size_match) / 3.0
            score += shape_score * 2.0
            
            # Proximity (weight 1) — centroid distance  
            in_grid_h = len(input_grid)
            in_grid_w = len(input_grid[0]) if input_grid else 1
            max_dist = (in_grid_h ** 2 + in_grid_w ** 2) ** 0.5
            if max_dist > 0:
                dist = ((in_obj["centroid"]["y"] - out_obj["centroid"]["y"]) ** 2 +
                        (in_obj["centroid"]["x"] - out_obj["centroid"]["x"]) ** 2) ** 0.5
                proximity = 1.0 - min(dist / max_dist, 1.0)
            else:
                proximity = 1.0
            score += proximity * 1.0
            
            if score > best_score:
                best_score = score
                best_match = out_obj
        
        if best_match is not None:
            matched_output_ids.add(best_match["id"])
            
            # Determine confidence (max possible score = 3 + 2 + 1 = 6)
            confidence = min(best_score / 6.0, 1.0)
            
            # Determine transformation type
            transform_type = _classify_transform(in_obj, best_match)
            
            correspondences.append({
                "input_obj": {
                    "color": in_obj["color"],
                    "size": in_obj["size"],
                    "position": in_obj["bounding_box"],
                    "dimensions": in_obj["dimensions"]
                },
                "output_obj": {
                    "color": best_match["color"],
                    "size": best_match["size"],
                    "position": best_match["bounding_box"],
                    "dimensions": best_match["dimensions"]
                },
                "confidence": round(confidence, 3),
                "transform_type": transform_type
            })
        else:
            # Input object was deleted (no match found)
            correspondences.append({
                "input_obj": {
                    "color": in_obj["color"],
                    "size": in_obj["size"],
                    "position": in_obj["bounding_box"],
                    "dimensions": in_obj["dimensions"]
                },
                "output_obj": None,
                "confidence": 1.0,
                "transform_type": "deleted"
            })
    
    # Check for created objects (output objects with no input match)
    for out_obj in output_objects:
        if out_obj["id"] not in matched_output_ids:
            correspondences.append({
                "input_obj": None,
                "output_obj": {
                    "color": out_obj["color"],
                    "size": out_obj["size"],
                    "position": out_obj["bounding_box"],
                    "dimensions": out_obj["dimensions"]
                },
                "confidence": 1.0,
                "transform_type": "created"
            })
    
    return correspondences


def _classify_transform(in_obj: Dict, out_obj: Dict) -> str:
    """Classify the type of transformation between matched objects."""
    color_changed = in_obj["color"] != out_obj["color"]
    position_changed = (in_obj["centroid"]["y"] != out_obj["centroid"]["y"] or
                       in_obj["centroid"]["x"] != out_obj["centroid"]["x"])
    size_changed = (in_obj["dimensions"]["width"] != out_obj["dimensions"]["width"] or
                   in_obj["dimensions"]["height"] != out_obj["dimensions"]["height"])
    
    if color_changed and not position_changed and not size_changed:
        return "recolored"
    elif position_changed and not size_changed and not color_changed:
        return "moved"
    elif size_changed:
        return "resized"
    elif color_changed and position_changed:
        return "moved_and_recolored"
    else:
        return "unchanged"
