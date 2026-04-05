import pytest
import numpy as np
import sys
import os

# Add parent directory to path so we can import app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import encode_input, psi, ternary, h7_process, N_VALS, phi, PI

def test_encode_input():
    # Test numeric
    assert encode_input(10) in N_VALS
    # Test string
    assert encode_input("test_string") in N_VALS
    # Test list
    assert encode_input([1, 2, 3]) in N_VALS

def test_psi():
    n = 1
    expected_psi = float(np.cos(PI * phi * n))
    assert np.isclose(psi(n), expected_psi)

def test_ternary():
    # Test epsilon = 0.25 (default)
    assert ternary(0.5) == 1
    assert ternary(-0.5) == -1
    assert ternary(0.1) == 0
    assert ternary(-0.1) == 0

def test_h7_process():
    result = h7_process("test_data")
    assert "n" in result
    assert "state_vector" in result
    assert "psi_value" in result
    assert "ternary_state" in result
    assert "binary_fw" in result
    assert "binary_bw" in result
    assert "label" in result
    
    n = result["n"]
    assert result["state_vector"] == (n, 7 - n)
    assert result["label"] in ["constructive", "equilibrium", "destructive"]
