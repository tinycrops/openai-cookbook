#!/usr/bin/env python3
"""
pdf_to_md.py: Extract text from a PDF and output as Markdown.
Usage:
    ./scripts/pdf_to_md.py input.pdf output.md
"""
import sys
import subprocess

def main():
    if len(sys.argv) != 3:
        print("Usage: {} input.pdf output.md".format(sys.argv[0]))
        sys.exit(1)
    pdf_path = sys.argv[1]
    md_path = sys.argv[2]
    try:
        # Use pdftotext to extract plain text with layout preserved
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        text = result.stdout.decode('utf-8', errors='ignore')
    except FileNotFoundError:
        print("Error: pdftotext utility not found. Install poppler-utils.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting text: {e.stderr.decode('utf-8', errors='ignore')}")
        sys.exit(1)
    # Write to Markdown file
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# {pdf_path}\n\n")
        # Optionally, split into pages or preserve as one block
        f.write(text)
    print(f"Extracted markdown written to {md_path}")

if __name__ == '__main__':
    main()