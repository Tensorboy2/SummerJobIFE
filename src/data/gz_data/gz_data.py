#!/usr/bin/env python3
import gzip
import os
import sys
from datetime import datetime

def read_gz_metadata(filepath):
    """Read and display metadata from a gzip file"""
    
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found")
        return
    
    try:
        with gzip.open(filepath, 'rb') as gz_file:
            # Get file stats
            file_stats = os.stat(filepath)
            compressed_size = file_stats.st_size
            modified_time = datetime.fromtimestamp(file_stats.st_mtime)
            
            print(f"=== GZ File Metadata: {filepath} ===")
            print(f"Compressed size: {compressed_size:,} bytes")
            print(f"Modified: {modified_time}")
            
            # Try to get original filename and timestamp from gzip header
            # Note: Not all gzip files have this info
            try:
                # Read first few bytes to check gzip header
                gz_file.seek(0)
                header = gz_file.read(10)
                
                if len(header) >= 10:
                    # Check magic number
                    if header[0:2] == b'\x1f\x8b':
                        print("Valid gzip file: ✓")
                        
                        # Check flags (byte 3)
                        flags = header[3]
                        print(f"Flags: {flags:08b}")
                        
                        # Extract timestamp (bytes 4-7)
                        timestamp = int.from_bytes(header[4:8], 'little')
                        if timestamp > 0:
                            orig_time = datetime.fromtimestamp(timestamp)
                            print(f"Original timestamp: {orig_time}")
                        
                        # Try to read some content to estimate uncompressed size
                        gz_file.seek(0)
                        sample = gz_file.read(1024)  # Read first 1KB
                        if sample:
                            print(f"Content starts with: {sample[:50]}...")
                            
                            # Try to estimate compression ratio
                            gz_file.seek(0)
                            try:
                                # Read all content to get uncompressed size
                                uncompressed = gz_file.read()
                                uncompressed_size = len(uncompressed)
                                ratio = compressed_size / uncompressed_size * 100
                                print(f"Uncompressed size: {uncompressed_size:,} bytes")
                                print(f"Compression ratio: {ratio:.1f}%")
                            except Exception as e:
                                print(f"Could not determine uncompressed size: {e}")
                    else:
                        print("Invalid gzip file: ✗")
                        
            except Exception as e:
                print(f"Error reading gzip header: {e}")
                
            # If it's a text file, try to show first few lines
            try:
                gz_file.seek(0)
                content = gz_file.read().decode('utf-8', errors='ignore')
                lines = content.split('\n')[:5]  # First 5 lines
                
                print("\n=== Content Preview ===")
                for i, line in enumerate(lines, 1):
                    if line.strip():
                        print(f"Line {i}: {line[:100]}{'...' if len(line) > 100 else ''}")
                        
            except Exception as e:
                print(f"Could not preview text content: {e}")
                
    except Exception as e:
        print(f"Error opening gzip file: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python gz_reader.py <path_to_gz_file>")
        print("Example: python gz_reader.py data.txt.gz")
        sys.exit(1)
    
    filepath = sys.argv[1]
    read_gz_metadata(filepath)

if __name__ == "__main__":
    main()