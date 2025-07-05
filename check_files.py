import os
from pathlib import Path

# Check current working directory
print(f"Current working directory: {os.getcwd()}")

# Check if artifacts folder exists
artifacts_path = Path("artifacts")
print(f"\nArtifacts folder exists: {artifacts_path.exists()}")

if artifacts_path.exists():
    print("\nFiles in artifacts folder:")
    for file in artifacts_path.iterdir():
        print(f"  - {file.name}")
else:
    print("Artifacts folder not found!")

# Check data folder
data_path = Path("data")
print(f"\nData folder exists: {data_path.exists()}")
if data_path.exists():
    print("\nFiles in data folder:")
    for file in data_path.iterdir():
        print(f"  - {file.name}")