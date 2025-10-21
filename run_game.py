#!/usr/bin/env python3
"""
Boxhead Game Runner - Simple launcher script
Choose between manual play or AI versions
"""

import os
import sys
import subprocess

def check_requirements():
    """Check if required packages are installed"""
    try:
        import pygame
        print("✓ pygame available")
    except ImportError:
        print("✗ pygame not found. Install with: pip install pygame")
        return False
    
    try:
        import stable_baselines3
        print("✓ stable-baselines3 available")
        return True
    except ImportError:
        print("! stable-baselines3 not found. Only manual play will be available.")
        print("  Install with: pip install stable-baselines3")
        return False

def main():
    print("Boxhead Game Launcher")
    print("=" * 40)
    
    # Check requirements
    has_sb3 = check_requirements()
    
    print("\nChoose game mode:")
    print("1. Manual Play (no AI required)")
    print("2. AI Versions (requires stable-baselines3)")
    print("3. Install requirements")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                print("\nStarting manual play...")
                subprocess.run([sys.executable, "manual_game.py"])
                break
            elif choice == "2":
                if has_sb3:
                    print("\nStarting AI game launcher...")
                    subprocess.run([sys.executable, "game_launcher.py"])
                    break
                else:
                    print("Error: stable-baselines3 not available. Please install it first.")
            elif choice == "3":
                print("\nInstalling requirements...")
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                print("Requirements installed. Please restart the launcher.")
                break
            elif choice == "4":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-4.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
