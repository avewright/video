"""
API Package

Contains the web API server and related components for the nameplate detector.
"""

from .server import create_app

__all__ = ["create_app"] 