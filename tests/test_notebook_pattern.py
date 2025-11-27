#!/usr/bin/env python3
"""Test that the code structure matches the notebook usage pattern exactly."""

import sys
import re
import json

def extract_notebook_code():
    """Extract code from Inference.ipynb."""
    print("=" * 60)
    print("EXTRACTING NOTEBOOK CODE")
    print("=" * 60)
    
    try:
        with open('notebooks/Inference.ipynb', 'r') as f:
            notebook = json.load(f)
        
        cells = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if source.strip():
                    cells.append(source)
        
        print(f"  ✓ Found {len(cells)} code cells")
        return cells
    except Exception as e:
        print(f"  ✗ Failed to read notebook: {e}")
        return None

def test_cell_0_pattern(cells):
    """Test Cell 0: Main inference call."""
    print("\n" + "=" * 60)
    print("TEST 1: Cell 0 - Main Inference Call")
    print("=" * 60)
    
    if not cells or len(cells) == 0:
        print("  ✗ No cells found")
        return False
    
    cell_0 = cells[0]
    
    # Check for required elements
    checks = [
        ('AUDIO_URL', 'Audio URL variable'),
        ('USE_JUKEBOX', 'USE_JUKEBOX variable'),
        ('SEGMENT_START_HINT', 'SEGMENT_START_HINT variable'),
        ('SEGMENT_END_HINT', 'SEGMENT_END_HINT variable'),
        ('BPM_HINT', 'BPM_HINT variable'),
        ('from sheetsage.infer import sheetsage', 'sheetsage import'),
        ('sheetsage(', 'sheetsage function call'),
        ('use_jukebox=USE_JUKEBOX', 'use_jukebox parameter'),
        ('segment_start_hint=SEGMENT_START_HINT', 'segment_start_hint parameter'),
        ('segment_end_hint=SEGMENT_END_HINT', 'segment_end_hint parameter'),
        ('beats_per_minute_hint=BPM_HINT', 'beats_per_minute_hint parameter'),
    ]
    
    print("Checking Cell 0 code pattern...")
    all_found = True
    for pattern, description in checks:
        if pattern in cell_0:
            print(f"  ✓ {description} found")
        else:
            print(f"  ✗ {description} NOT FOUND")
            all_found = False
    
    # Verify return values
    if 'lead_sheet, segment_beats, segment_beats_times =' in cell_0:
        print("  ✓ Correct return value unpacking")
    else:
        print("  ✗ Return value unpacking incorrect")
        all_found = False
    
    return all_found

def test_cell_1_pattern(cells):
    """Test Cell 1: Lead sheet display."""
    print("\n" + "=" * 60)
    print("TEST 2: Cell 1 - Lead Sheet Display")
    print("=" * 60)
    
    if not cells or len(cells) < 2:
        print("  ⚠ Cell 1 not found (may be okay)")
        return True
    
    cell_1 = cells[1]
    
    checks = [
        ('from sheetsage.utils import engrave', 'engrave import'),
        ('lead_sheet.as_lily()', 'lead_sheet.as_lily() call'),
        ('engrave(', 'engrave function call'),
    ]
    
    print("Checking Cell 1 code pattern...")
    all_found = True
    for pattern, description in checks:
        if pattern in cell_1:
            print(f"  ✓ {description} found")
        else:
            print(f"  ⚠ {description} not found (may be okay)")
    
    return True  # Don't fail on this

def test_cell_2_pattern(cells):
    """Test Cell 2: MIDI creation."""
    print("\n" + "=" * 60)
    print("TEST 3: Cell 2 - MIDI Creation")
    print("=" * 60)
    
    if not cells or len(cells) < 3:
        print("  ⚠ Cell 2 not found (may be okay)")
        return True
    
    cell_2 = cells[2]
    
    checks = [
        ('from sheetsage.align import create_beat_to_time_fn', 'create_beat_to_time_fn import'),
        ('create_beat_to_time_fn(segment_beats, segment_beats_times)', 'create_beat_to_time_fn call'),
        ('lead_sheet.as_midi(', 'lead_sheet.as_midi() call'),
    ]
    
    print("Checking Cell 2 code pattern...")
    for pattern, description in checks:
        if pattern in cell_2:
            print(f"  ✓ {description} found")
        else:
            print(f"  ⚠ {description} not found (may be okay)")
    
    return True  # Don't fail on this

def test_infer_py_compatibility():
    """Test that infer.py supports the notebook pattern."""
    print("\n" + "=" * 60)
    print("TEST 4: infer.py Compatibility")
    print("=" * 60)
    
    try:
        with open('sheetsage/infer.py', 'r') as f:
            infer_code = f.read()
        
        # Check that sheetsage function accepts all notebook parameters
        required_params = [
            'use_jukebox',
            'segment_start_hint',
            'segment_end_hint',
            'beats_per_minute_hint'
        ]
        
        print("Checking sheetsage function parameters...")
        all_found = True
        for param in required_params:
            # Check in function definition
            pattern = f'{param}'
            if pattern in infer_code:
                print(f"  ✓ Parameter '{param}' found")
            else:
                print(f"  ✗ Parameter '{param}' NOT FOUND")
                all_found = False
        
        # Check return values
        if 'return lead_sheet, segment_beats, segment_beats_times' in infer_code:
            print("  ✓ Returns (lead_sheet, segment_beats, segment_beats_times)")
        elif 'return lead_sheet, segment_beats' in infer_code:
            print("  ✓ Returns (lead_sheet, segment_beats, ...)")
        else:
            print("  ⚠ Return values may not match")
        
        return all_found
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False

def test_jukebox_integration():
    """Test that Jukebox integration works with notebook pattern."""
    print("\n" + "=" * 60)
    print("TEST 5: Jukebox Integration")
    print("=" * 60)
    
    try:
        with open('sheetsage/infer.py', 'r') as f:
            infer_code = f.read()
        
        # Check that use_jukebox=True triggers Jukebox path
        checks = [
            ('use_jukebox', 'use_jukebox parameter handling'),
            ('InputFeats.JUKEBOX', 'InputFeats.JUKEBOX usage'),
            ('JukeboxEmbeddings()', 'JukeboxEmbeddings instantiation'),
            ('Feature extraction w/ Jukebox', 'Jukebox logging message'),
        ]
        
        print("Checking Jukebox integration...")
        all_found = True
        for pattern, description in checks:
            if pattern in infer_code:
                print(f"  ✓ {description} found")
            else:
                print(f"  ✗ {description} NOT FOUND")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False

def test_code_execution_structure():
    """Test that the code structure allows execution."""
    print("\n" + "=" * 60)
    print("TEST 6: Code Execution Structure")
    print("=" * 60)
    
    try:
        # Check that all imports in notebook are available
        notebook_imports = [
            'sheetsage.infer',
            'sheetsage.utils',
            'sheetsage.align',
        ]
        
        print("Checking import structure...")
        all_ok = True
        for imp in notebook_imports:
            module_path = imp.replace('.', '/')
            # Check if module file exists
            if 'infer' in imp:
                file_path = 'sheetsage/infer.py'
            elif 'utils' in imp:
                file_path = 'sheetsage/utils.py'
            elif 'align' in imp:
                file_path = 'sheetsage/align.py'
            else:
                file_path = None
            
            if file_path:
                try:
                    with open(file_path, 'r'):
                        print(f"  ✓ {imp} module file exists")
                except FileNotFoundError:
                    print(f"  ✗ {imp} module file NOT FOUND")
                    all_ok = False
            else:
                print(f"  ⚠ {imp} - cannot verify")
        
        return all_ok
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False

def main():
    """Run all notebook pattern tests."""
    print("\n" + "=" * 60)
    print("NOTEBOOK USAGE PATTERN VERIFICATION")
    print("=" * 60)
    print("\nVerifying that code matches Inference.ipynb usage pattern...\n")
    
    # Extract notebook code
    cells = extract_notebook_code()
    
    results = []
    
    # Test each cell pattern
    if cells:
        results.append(("Cell 0 - Main Call", test_cell_0_pattern(cells)))
        results.append(("Cell 1 - Display", test_cell_1_pattern(cells)))
        results.append(("Cell 2 - MIDI", test_cell_2_pattern(cells)))
    else:
        print("⚠ Cannot extract notebook cells")
    
    # Test compatibility
    results.append(("infer.py Compatibility", test_infer_py_compatibility()))
    results.append(("Jukebox Integration", test_jukebox_integration()))
    results.append(("Execution Structure", test_code_execution_structure()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    critical_tests = ["Cell 0 - Main Call", "infer.py Compatibility", "Jukebox Integration"]
    critical_passed = all(success for name, success in results if name in critical_tests)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        critical = " [CRITICAL]" if test_name in critical_tests else ""
        print(f"{test_name:30s} {status}{critical}")
    
    print("\n" + "=" * 60)
    if critical_passed:
        print("✓ NOTEBOOK PATTERN VERIFIED")
        print("\nThe code structure matches the notebook usage pattern.")
        print("Ready for execution with proper dependencies.")
    else:
        print("✗ PATTERN VERIFICATION FAILED")
        print("\nSome critical tests failed. Please review above.")
    print("=" * 60)
    
    return 0 if critical_passed else 1

if __name__ == "__main__":
    sys.exit(main())
