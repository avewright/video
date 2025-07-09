#!/usr/bin/env python3
"""
Command Line Interface for Nameplate Detector

Provides a unified CLI for all nameplate detector operations.
"""

import argparse
import sys
import logging
from pathlib import Path

from .config.settings import get_settings
from .utils.helpers import setup_logging
from .api.server import main as api_main


def create_parser():
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Nameplate Detector - Real-time video streaming with nameplate detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s api                          # Start API server
  %(prog)s api --port 8080              # Start API server on port 8080
  %(prog)s predict image.jpg            # Predict nameplate in image
  %(prog)s config                       # Show current configuration
  %(prog)s --version                    # Show version information
        """
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start API server")
    api_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    api_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    api_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict nameplate in image")
    predict_parser.add_argument(
        "image_path",
        type=str,
        help="Path to image file"
    )
    predict_parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model file"
    )
    predict_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for prediction"
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        help="Output file for results"
    )
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration")
    config_parser.add_argument(
        "--format",
        choices=["json", "yaml", "env"],
        default="json",
        help="Output format"
    )
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage"
    )
    test_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    return parser


def cmd_api(args):
    """Handle API command."""
    import uvicorn
    from .api.server import app
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower()
    )


def cmd_predict(args):
    """Handle predict command."""
    from .models.classifier import NameplateClassifierLoader
    from .utils.helpers import validate_image, format_confidence
    import json
    
    # Validate input
    if not Path(args.image_path).exists():
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    if not validate_image(args.image_path):
        print(f"Error: Invalid image file: {args.image_path}")
        sys.exit(1)
    
    # Load classifier
    try:
        classifier = NameplateClassifierLoader(model_path=args.model_path)
        has_nameplate, confidence = classifier.predict(args.image_path)
        
        result = {
            "image_path": args.image_path,
            "has_nameplate": has_nameplate,
            "confidence": confidence,
            "confidence_formatted": format_confidence(confidence),
            "threshold": args.confidence_threshold,
            "passes_threshold": confidence >= args.confidence_threshold
        }
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)


def cmd_config(args):
    """Handle config command."""
    import json
    
    settings = get_settings()
    config_dict = settings.to_dict()
    
    if args.format == "json":
        print(json.dumps(config_dict, indent=2))
    elif args.format == "yaml":
        try:
            import yaml
            print(yaml.dump(config_dict, default_flow_style=False))
        except ImportError:
            print("PyYAML not installed. Install with: pip install PyYAML")
            sys.exit(1)
    elif args.format == "env":
        for key, value in config_dict.items():
            print(f"{key.upper()}={value}")


def cmd_test(args):
    """Handle test command."""
    import subprocess
    
    cmd = ["python", "-m", "pytest"]
    
    if args.coverage:
        cmd.extend(["--cov=nameplate_detector", "--cov-report=html"])
    
    if args.verbose:
        cmd.append("-v")
    
    cmd.append("tests/")
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    
    # Handle commands
    if args.command == "api":
        cmd_api(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "config":
        cmd_config(args)
    elif args.command == "test":
        cmd_test(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 