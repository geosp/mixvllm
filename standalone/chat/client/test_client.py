#!/usr/bin/env python3
"""
Simple test script for the refactored chat client.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from client import ChatClient, ChatConfig

def test_imports():
    """Test that all imports work."""
    print("✓ Imports successful")

def test_config():
    """Test configuration creation."""
    config = ChatConfig(
        base_url="http://localhost:8000",
        model="test-model",
        enable_mcp=False,
        debug=False
    )
    assert config.base_url == "http://localhost:8000"
    assert config.model == "test-model"
    print("✓ Configuration test passed")

def test_config_from_args():
    """Test configuration from args."""
    class MockArgs:
        base_url = "http://test:8000"
        model = "test-model"
        enable_mcp = True
        debug = True
        mcp_config = None
        temperature = 0.8
        max_tokens = 256
        stream = True

    args = MockArgs()
    config = ChatConfig.from_args(args)
    assert config.base_url == "http://test:8000"
    assert config.temperature == 0.8
    print("✓ Configuration from args test passed")

if __name__ == "__main__":
    print("Testing refactored chat client...")
    test_imports()
    test_config()
    test_config_from_args()
    print("✓ All tests passed!")