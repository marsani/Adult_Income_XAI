import sys
import importlib.metadata
import importlib_metadata

print(f"Python version: {sys.version}")
print(f"importlib_metadata has packages_distributions: {hasattr(importlib_metadata, 'packages_distributions')}")

if not hasattr(importlib.metadata, 'packages_distributions'):
    print("Monkeypatching importlib.metadata.packages_distributions...")
    importlib.metadata.packages_distributions = importlib_metadata.packages_distributions

try:
    import google.generativeai as genai
    print("google.generativeai imported successfully AFTER monkeypatch")
except Exception as e:
    print(f"Error importing google.generativeai: {e}")
    import traceback
    traceback.print_exc()
