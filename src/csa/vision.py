import numpy as np
from scipy.ndimage import label, find_objects
from typing import List, Dict, Any, Tuple

class GridObject:
    """Represents a discrete contiguous shape on the ARC grid."""
    def __init__(self, obj_id: int, color: int, mask: np.ndarray, bounding_box: tuple):
        self.obj_id = obj_id
        self.color = color
        self.mask = mask # Boolean mask of the object
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
        """
        Takes raw 2D grid, outputs a heavily structured symbolic graph.
        """
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
        
        # Iterate over all possible colors (0-9 for ARC)
        for color in range(10):
            if color == self.bg_color:
                continue
                
            color_mask = (grid == color).astype(int)
            if np.sum(color_mask) == 0:
                continue
                
            # Use scipy to label connected components (diagonal connections not included default, 
            # usually setting structure to a 3x3 array of 1s includes diagonals)
            structure = np.ones((3, 3), dtype=int) 
            labeled_array, num_features = label(color_mask, structure=structure)
            
            # Find bounding boxes for each labeled feature
            slices = find_objects(labeled_array)
            
            for i, slc in enumerate(slices):
                if slc is not None:
                    # Create isolated mask for just THIS specific object
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
        """
        Derives high-level spatial rules from the extracted objects.
        For example: Object 1 is ABOVE Object 2.
        """
        relations = []
        
        for i, obj_a in enumerate(objects):
            for j, obj_b in enumerate(objects):
                if i == j:
                    continue
                    
                # A is completely above B
                if obj_a.bottom < obj_b.top:
                    relations.append({"subject_id": obj_a.obj_id, "relation": "is_above", "target_id": obj_b.obj_id})
                
                # A is completely below B
                if obj_a.top > obj_b.bottom:
                    relations.append({"subject_id": obj_a.obj_id, "relation": "is_below", "target_id": obj_b.obj_id})
                    
                # A is completely to the left of B
                if obj_a.right < obj_b.left:
                    relations.append({"subject_id": obj_a.obj_id, "relation": "is_left_of", "target_id": obj_b.obj_id})
                    
                # A is completely to the right of B
                if obj_a.left > obj_b.right:
                    relations.append({"subject_id": obj_a.obj_id, "relation": "is_right_of", "target_id": obj_b.obj_id})
                    
                # Enclosure (A is completely inside B's bounding box)
                # Note: true enclosure requires mask checking, but bounding box is a fast heuristic
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
