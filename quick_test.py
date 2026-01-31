"""
Quick Test for ARC-AGI Solver

Run this to verify all components work:
    python quick_test.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all imports work."""
    print("Testing imports...")
    
    try:
        from src.arc.enhanced_dsl import enhanced_dsl_registry
        print(f"  ✓ Enhanced DSL: {len(enhanced_dsl_registry)} primitives")
    except Exception as e:
        print(f"  ✗ Enhanced DSL failed: {e}")
        return False
    
    try:
        from src.arc.object_detector import ObjectDetector, analyze_grid
        print("  ✓ Object Detector imported")
    except Exception as e:
        print(f"  ✗ Object Detector failed: {e}")
        return False
    
    try:
        from src.arc.advanced_patterns import AdvancedPatternDetector
        print("  ✓ Advanced Pattern Detector imported")
    except Exception as e:
        print(f"  ✗ Advanced Pattern Detector failed: {e}")
        return False
    
    try:
        from src.arc.super_reasoning import SuperReasoningEngine
        print("  ✓ Super Reasoning Engine imported")
    except Exception as e:
        print(f"  ✗ Super Reasoning Engine failed: {e}")
        return False
    
    return True


def test_dsl():
    """Test DSL primitives."""
    import numpy as np
    from src.arc.enhanced_dsl import enhanced_dsl_registry
    
    print("\nTesting DSL primitives...")
    
    # Test grid
    grid = np.array([
        [1, 2, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    # Test some primitives
    tests = [
        ("rotate_cw", lambda g: np.rot90(g, -1)),
        ("reflect_h", lambda g: np.fliplr(g)),
        ("crop", lambda g: g),  # Will test shape
    ]
    
    passed = 0
    for name, expected_fn in tests:
        if name in enhanced_dsl_registry:
            try:
                result = enhanced_dsl_registry[name](grid)
                print(f"  ✓ {name} works")
                passed += 1
            except Exception as e:
                print(f"  ✗ {name} failed: {e}")
        else:
            print(f"  ? {name} not found")
    
    print(f"  Passed: {passed}/{len(tests)}")
    return True


def test_object_detector():
    """Test object detection."""
    import numpy as np
    from src.arc.object_detector import ObjectDetector, analyze_grid
    
    print("\nTesting Object Detector...")
    
    grid = np.array([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 3, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    
    detector = ObjectDetector()
    objects = detector.detect_objects(grid)
    print(f"  ✓ Detected {len(objects)} objects")
    
    for obj in objects:
        print(f"    - Object {obj.id}: color={obj.color}, size={obj.size}, shape={obj.shape_type.value}")
    
    analysis = analyze_grid(grid)
    print(f"  ✓ Grid analysis: {analysis.num_objects} objects, {len(analysis.unique_colors)} colors")
    
    return True


def test_pattern_detector():
    """Test advanced pattern detection."""
    import numpy as np
    from src.arc.advanced_patterns import AdvancedPatternDetector
    
    print("\nTesting Advanced Pattern Detector...")
    
    # Simple scaling pattern
    examples = [
        (np.array([[1]]), np.array([[1, 1], [1, 1]])),
        (np.array([[2]]), np.array([[2, 2], [2, 2]])),
    ]
    
    detector = AdvancedPatternDetector()
    patterns = detector.detect_all_patterns(examples)
    
    print(f"  ✓ Found {len(patterns)} patterns")
    for p in patterns[:3]:
        print(f"    - {p.description} (confidence: {p.confidence:.2f})")
    
    return True


def test_reasoning():
    """Test reasoning engine."""
    import numpy as np
    from src.arc.super_reasoning import SuperReasoningEngine
    
    print("\nTesting Super Reasoning Engine...")
    
    # Simple rotation task
    examples = [
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])),
        (np.array([[2, 0], [0, 0]]), np.array([[0, 2], [0, 0]])),
    ]
    test_input = np.array([[3, 0], [0, 0]])
    expected = np.array([[0, 3], [0, 0]])
    
    engine = SuperReasoningEngine()
    result = engine.solve(examples, test_input, "test_rotate")
    
    print(f"  Strategy: {result.strategy.value}")
    print(f"  Description: {result.description}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Success: {result.success}")
    
    if result.prediction is not None:
        if np.array_equal(result.prediction, expected):
            print("  ✓ Prediction correct!")
        else:
            print("  ✗ Prediction incorrect")
            print(f"    Expected:\n{expected}")
            print(f"    Got:\n{result.prediction}")
    
    return result.success


def run_mini_benchmark():
    """Run a mini benchmark."""
    import numpy as np
    from src.arc.super_reasoning import SuperReasoningEngine
    
    print("\n" + "="*50)
    print("MINI BENCHMARK")
    print("="*50)
    
    # Test cases
    test_cases = [
        {
            'name': 'Horizontal Reflection',
            'train': [
                (np.array([[1,2,0],[0,0,0]]), np.array([[0,2,1],[0,0,0]])),
                (np.array([[3,0,0],[4,0,0]]), np.array([[0,0,3],[0,0,4]])),
            ],
            'test_input': np.array([[5,6,0],[0,0,0]]),
            'test_output': np.array([[0,6,5],[0,0,0]]),
        },
        {
            'name': '90° Rotation',
            'train': [
                (np.array([[1,0],[0,0]]), np.array([[0,1],[0,0]])),
                (np.array([[2,3],[0,0]]), np.array([[0,2],[0,3]])),
            ],
            'test_input': np.array([[4,0],[5,0]]),
            'test_output': np.array([[5,4],[0,0]]),
        },
        {
            'name': 'Scale 2x',
            'train': [
                (np.array([[1]]), np.array([[1,1],[1,1]])),
                (np.array([[2,0],[0,3]]), np.array([[2,2,0,0],[2,2,0,0],[0,0,3,3],[0,0,3,3]])),
            ],
            'test_input': np.array([[5]]),
            'test_output': np.array([[5,5],[5,5]]),
        },
        {
            'name': 'Keep Largest Object',
            'train': [
                (np.array([[1,0,2],[0,2,2],[0,2,2]]), np.array([[0,0,2],[0,2,2],[0,2,2]])),
            ],
            'test_input': np.array([[1,1,0],[0,0,3],[0,0,0]]),
            'test_output': np.array([[1,1,0],[0,0,0],[0,0,0]]),
        },
        {
            'name': 'Crop to Content',
            'train': [
                (np.array([[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]]), np.array([[1,1],[1,1]])),
                (np.array([[0,0,0],[0,5,0],[0,0,0]]), np.array([[5]])),
            ],
            'test_input': np.array([[0,0,0,0,0],[0,0,3,3,0],[0,0,3,3,0],[0,0,0,0,0]]),
            'test_output': np.array([[3,3],[3,3]]),
        },
    ]
    
    engine = SuperReasoningEngine()
    passed = 0
    
    for tc in test_cases:
        result = engine.solve(tc['train'], tc['test_input'], tc['name'])
        
        if result.success and result.prediction is not None:
            if np.array_equal(result.prediction, tc['test_output']):
                print(f"  ✓ {tc['name']}: PASSED ({result.strategy.value})")
                passed += 1
            else:
                print(f"  ✗ {tc['name']}: Wrong answer")
        else:
            print(f"  ✗ {tc['name']}: Failed to solve")
    
    accuracy = passed / len(test_cases) * 100
    print(f"\nMini Benchmark: {passed}/{len(test_cases)} ({accuracy:.0f}%)")
    
    return passed == len(test_cases)


def main():
    """Run all tests."""
    print("="*50)
    print("ARC-AGI SOLVER QUICK TEST")
    print("="*50)
    
    all_passed = True
    
    if not test_imports():
        print("\n❌ Import tests failed!")
        return False
    
    if not test_dsl():
        all_passed = False
    
    if not test_object_detector():
        all_passed = False
    
    if not test_pattern_detector():
        all_passed = False
    
    if not test_reasoning():
        all_passed = False
    
    run_mini_benchmark()
    
    print("\n" + "="*50)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("⚠ SOME TESTS FAILED")
    print("="*50)
    
    return all_passed


if __name__ == "__main__":
    main()
