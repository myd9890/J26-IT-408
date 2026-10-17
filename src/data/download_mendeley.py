"""
Compatibility shim: logic moved to scripts/data/download_mendeley.py
"""

from scripts.data.download_mendeley import download_file, extract_zip, main

if __name__ == "__main__":
    main()
