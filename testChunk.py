"""
final_chunk_viewer.py

FIXED PARSER - Xử lý đúng format thực tế của file
"""

import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunkers.recursive_chunker import RecursiveChunker
from chunkers.page_aware_chunker import PageAwareChunker
from chunkers.semantic_chunker import SemanticChunker


def parse_pages_correctly(file_path):
    """
    Parse đúng format:

    ================================================================================
    PAGE 1 | GROQ
    ================================================================================
    [Text của page 1]
    ================================================================================
    PAGE 2 | GROQ
    ================================================================================
    [Text của page 2]
    """
    content = Path(file_path).read_text(encoding='utf-8')

    separator = '='*80

    # Split by separator
    blocks = content.split(separator)

    pages = []
    current_page_num = None

    for i, block in enumerate(blocks):
        if not block.strip():
            continue

        lines = block.strip().split('\n')

        # Check if this block contains PAGE marker
        page_match = None
        for line in lines:
            match = re.match(r'PAGE\s+(\d+)', line)
            if match:
                page_match = match
                break

        if page_match:
            # This block has PAGE marker
            page_num = int(page_match.group(1))

            # Text of this page is in NEXT block(s)
            # Save current page_num for next iterations
            current_page_num = page_num

            # Initialize page
            pages.append({'page_num': page_num, 'text': ''})

        else:
            # This block is content
            text = block.strip()

            if text and current_page_num is not None:
                # Find the page dict and append text
                for page in pages:
                    if page['page_num'] == current_page_num:
                        if page['text']:
                            page['text'] += '\n\n' + text
                        else:
                            page['text'] = text
                        break

    # Remove empty pages
    pages = [p for p in pages if p['text'].strip()]

    return pages


def show_chunks(chunks, name, max_show=3):
    """Display chunks preview"""
    if not chunks:
        print(f"❌ No chunks for {name}!")
        return

    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    print(f"Total: {len(chunks)} chunks")

    sizes = [c.get('chunk_size', len(c['text'])) for c in chunks]
    print(f"Avg: {sum(sizes)/len(sizes):.0f} chars | Range: {min(sizes)}-{max(sizes)}")

    print(f"\nFirst {min(max_show, len(chunks))} chunks:")
    print("─"*80)

    for i, chunk in enumerate(chunks[:max_show], 1):
        page = chunk.get('page_num', '?')
        size = chunk.get('chunk_size', len(chunk['text']))
        ctype = chunk.get('chunk_type', 'N/A')

        print(f"\nChunk {i}:")
        print(f"  📄 Page {page} | 📏 {size} chars | 🏷️  {ctype}")

        # Preview first 100 chars
        text_preview = chunk['text'][:100].replace('\n', ' ')
        print(f"  📝 {text_preview}...")


def save_to_file(chunks, filename, strategy_name):
    """Save chunks to file"""
    Path("output").mkdir(exist_ok=True)

    filepath = f"output/{filename}"

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"CHUNKS - {strategy_name}\n")
        f.write(f"="*80 + "\n\n")
        f.write(f"Total chunks: {len(chunks)}\n\n")

        if chunks:
            sizes = [c.get('chunk_size', len(c['text'])) for c in chunks]
            f.write(f"Average size: {sum(sizes)/len(sizes):.0f} chars\n")
            f.write(f"Min size: {min(sizes)} chars\n")
            f.write(f"Max size: {max(sizes)} chars\n\n")

        f.write("="*80 + "\n\n")

        for i, chunk in enumerate(chunks, 1):
            f.write("="*80 + "\n")
            f.write(f"CHUNK {i}/{len(chunks)}\n")
            f.write("="*80 + "\n")
            f.write(f"Page: {chunk.get('page_num', 'N/A')}\n")
            f.write(f"Size: {chunk.get('chunk_size', len(chunk['text']))} chars\n")
            f.write(f"Type: {chunk.get('chunk_type', 'N/A')}\n")

            if 'section_heading' in chunk:
                f.write(f"Section: {chunk['section_heading']}\n")

            f.write("\n")
            f.write(chunk['text'])
            f.write("\n\n")

    return filepath


def main():
    print("\n" + "="*80)
    print("FINAL CHUNK VIEWER - WORKING VERSION")
    print("="*80)

    # Input file
    file_input = input("\n📂 File (Enter = output/after_llm.txt): ").strip()
    if not file_input:
        file_input = "output/after_llm.txt"

    if not Path(file_input).exists():
        print(f"❌ File not found: {file_input}")
        return

    file_size = Path(file_input).stat().st_size
    print(f"✅ Found: {file_input} ({file_size:,} bytes)")

    # Parse pages
    print(f"\n{'='*80}")
    print("PARSING PAGES")
    print("="*80)
    print("Loading...")

    pages = parse_pages_correctly(file_input)

    if not pages:
        print("❌ Could not parse any pages!")
        print("\nPlease check file format.")
        return

    print(f"✅ Parsed {len(pages)} pages")

    # Show sample
    if pages:
        sample = pages[0]
        print(f"\n📄 Sample Page {sample['page_num']}:")
        print(f"   Length: {len(sample['text'])} chars")
        print(f"   Preview: {sample['text'][:120]}...")

    # Choose strategy
    print(f"\n{'='*80}")
    print("CHOOSE CHUNKING STRATEGY")
    print("="*80)
    print("  1. Recursive (512 tokens) - Balanced chunks")
    print("  2. Page-Aware (2000 chars) - Preserve pages")
    print("  3. Semantic - Detect sections")
    print("  4. ALL - Compare all 3")

    choice = input("\nChoice (1-4, default=4): ").strip() or "4"

    # Map choices
    strategies_map = {
        "1": [("Recursive", RecursiveChunker(chunk_size=512), "chunks_recursive.txt")],
        "2": [("Page-Aware", PageAwareChunker(max_chars=2000), "chunks_page_aware.txt")],
        "3": [("Semantic", SemanticChunker(detect_headings=True), "chunks_semantic.txt")],
        "4": [
            ("Recursive", RecursiveChunker(chunk_size=512), "chunks_recursive.txt"),
            ("Page-Aware", PageAwareChunker(max_chars=2000), "chunks_page_aware.txt"),
            ("Semantic", SemanticChunker(detect_headings=True), "chunks_semantic.txt")
        ]
    }

    selected = strategies_map.get(choice, strategies_map["4"])

    # Process each strategy
    results = []

    for name, chunker, filename in selected:
        print(f"\n{'='*80}")
        print(f"PROCESSING: {name}")
        print("="*80)

        try:
            chunks = chunker.chunk_document(pages)

            if not chunks:
                print(f"⚠️  No chunks created!")
                continue

            print(f"✅ Created {len(chunks)} chunks")

            # Show preview
            show_chunks(chunks, name)

            # Save to file
            saved_path = save_to_file(chunks, filename, name)
            print(f"\n💾 Saved to: {saved_path}")

            results.append((name, chunks))

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print("="*80)
        print()
        print(f"{'Strategy':<15} {'Chunks':<10} {'Avg Size':<12} {'Range':<15}")
        print("─"*52)

        for name, chunks in results:
            sizes = [c.get('chunk_size', len(c['text'])) for c in chunks]
            avg = sum(sizes) / len(sizes)
            range_str = f"{min(sizes)}-{max(sizes)}"

            print(f"{name:<15} {len(chunks):<10} {avg:<12.0f} {range_str:<15}")

    # Done
    print(f"\n{'='*80}")
    print("✅ COMPLETED!")
    print("="*80)
    print()
    print("📁 Output files in: output/")

    if len(selected) == 1:
        print(f"   - {selected[0][2]}")
    else:
        for _, _, filename in selected:
            print(f"   - {filename}")

    print()
    print("💡 Open files to see all chunks in detail!")
    print()


if __name__ == "__main__":
    main()