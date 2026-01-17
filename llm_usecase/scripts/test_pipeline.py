#!/usr/bin/env python3
"""
Test script to verify the LLM case study pipeline works with a small subset of data.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path

def create_test_data():
    """Create a small test dataset."""
    
    # Create test QA data (first 5 queries)
    qa_file = 'data/qa/qas.jsonl'
    test_qa_file = 'data/qa/test_qas.jsonl'
    
    with open(qa_file, 'r') as f:
        qa_data = [json.loads(line) for line in f][:5]  # First 5 queries
    
    with open(test_qa_file, 'w') as f:
        for item in qa_data:
            f.write(json.dumps(item) + '\n')
    
    # Create test table data (first 100 rows)
    table_file = 'data/tables/WDC_product_for_cls.jsonl'
    test_table_file = 'data/tables/test_WDC_product_for_cls.jsonl'
    
    with open(table_file, 'r') as f:
        table_data = [json.loads(line) for line in f][:100]  # First 100 rows
    
    with open(test_table_file, 'w') as f:
        for item in table_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created test data:")
    print(f"  - QA: {len(qa_data)} queries -> {test_qa_file}")
    print(f"  - Tables: {len(table_data)} rows -> {test_table_file}")
    
    return test_qa_file, test_table_file

def test_build_index():
    """Test building index with BERT."""
    print("\n=== Testing Index Building ===")
    
    import subprocess
    import sys
    
    cmd = [
        sys.executable, 'scripts/build_index.py',
        '--data_dir', 'data',
        '--model_type', 'bert',
        '--output_dir', 'index/test_bert'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Index building successful")
        return True
    else:
        print(f"✗ Index building failed:")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False

def test_retrieve():
    """Test retrieval."""
    print("\n=== Testing Retrieval ===")
    
    import subprocess
    import sys
    
    # First create a small test index
    if not test_build_index():
        return False
    
    cmd = [
        sys.executable, 'scripts/retrieve.py',
        '--index_dir', 'index/test_bert',
        '--query_file', 'data/qa/test_qas.jsonl',
        '--model_type', 'bert',
        '--top_k', '3'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Retrieval successful")
        # Save test results
        with open('runs/test_retrieved.jsonl', 'w') as f:
            f.write(result.stdout)
        return True
    else:
        print(f"✗ Retrieval failed:")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False

def test_evaluation():
    """Test evaluation script."""
    print("\n=== Testing Evaluation ===")
    
    import subprocess
    import sys
    
    # Create mock predictions for testing
    mock_predictions = []
    with open('data/qa/test_qas.jsonl', 'r') as f:
        for line in f:
            qa_item = json.loads(line)
            mock_predictions.append({
                'query_id': qa_item['id'],
                'answer': qa_item['answer']  # Use ground truth as mock prediction
            })
    
    with open('runs/test_predictions.jsonl', 'w') as f:
        for item in mock_predictions:
            f.write(json.dumps(item) + '\n')
    
    cmd = [
        sys.executable, 'scripts/evaluate.py',
        '--predictions_file', 'runs/test_predictions.jsonl',
        '--labels_file', 'data/qa/test_qas.jsonl',
        '--analyze_by_type',
        '--output_file', 'runs/test_evaluation.json'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Evaluation successful")
        print("Evaluation output:")
        print(result.stdout)
        return True
    else:
        print(f"✗ Evaluation failed:")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False

def cleanup_test_files():
    """Clean up test files."""
    test_files = [
        'data/qa/test_qas.jsonl',
        'data/tables/test_WDC_product_for_cls.jsonl',
        'runs/test_retrieved.jsonl',
        'runs/test_predictions.jsonl',
        'runs/test_evaluation.json'
    ]
    
    test_dirs = [
        'index/test_bert'
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")
    
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Removed: {dir_path}")

def main():
    """Run the complete test pipeline."""
    print("=== LLM Case Study Pipeline Test ===")
    
    try:
        # Create test data
        create_test_data()
        
        # Test individual components
        tests_passed = 0
        total_tests = 3
        
        if test_build_index():
            tests_passed += 1
        
        if test_retrieve():
            tests_passed += 1
        
        if test_evaluation():
            tests_passed += 1
        
        print(f"\n=== Test Results ===")
        print(f"Tests passed: {tests_passed}/{total_tests}")
        
        if tests_passed == total_tests:
            print("✓ All tests passed! The pipeline is ready to run.")
        else:
            print("✗ Some tests failed. Please check the errors above.")
        
    finally:
        # Clean up test files
        print("\n=== Cleaning up test files ===")
        cleanup_test_files()

if __name__ == '__main__':
    main()
