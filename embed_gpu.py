"""
embed_gpu.py

GPU-OPTIMIZED EMBEDDING
Run from project root: python embed_gpu.py
"""

import sys
from pathlib import Path
import numpy as np
import json
import torch
from time import time

# Add current dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedders.single_embedder import SingleEmbedder
from embedders.vietnamese_embedder import VietnameseEmbedder


def check_gpu():
    """Check GPU"""
    print("GPU CHECK")
    print("=" * 80)

    if torch.cuda.is_available():
        print("✅ CUDA available!")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        print(f"   CUDA version: {torch.version.cuda}")
        return True
    else:
        print("❌ No CUDA GPU found")
        print("   Will use CPU (slower)")
        return False


def load_chunks_from_file(filepath):
    """Load chunks from txt file"""
    content = Path(filepath).read_text(encoding='utf-8')

    chunks = []
    current_chunk = None
    in_text = False
    text_lines = []

    for line in content.split('\n'):
        if line.startswith('CHUNK '):
            if current_chunk and text_lines:
                current_chunk['text'] = '\n'.join(text_lines).strip()
                chunks.append(current_chunk)

            current_chunk = {}
            text_lines = []
            in_text = False

        elif line.startswith('Page: '):
            try:
                current_chunk['page_num'] = int(line.split('Page: ')[1].strip())
            except:
                pass

        elif line.startswith('Size: '):
            try:
                current_chunk['chunk_size'] = int(line.split('Size: ')[1].split()[0])
            except:
                pass

        elif line.startswith('Type: '):
            current_chunk['chunk_type'] = line.split('Type: ')[1].strip()

        elif line.strip() == '' and not in_text and current_chunk:
            in_text = True

        elif in_text and not line.startswith('==='):
            text_lines.append(line)

    if current_chunk and text_lines:
        current_chunk['text'] = '\n'.join(text_lines).strip()
        chunks.append(current_chunk)

    return chunks


def main():
    print("\n" + "=" * 80)
    print("GPU-OPTIMIZED EMBEDDING")
    print("=" * 80 + "\n")

    # Check GPU
    has_gpu = check_gpu()
    print()

    # Input
    print("INPUT")
    print("=" * 80)
    chunks_file = input("Chunks file (Enter = output/chunks_recursive.txt): ").strip()
    if not chunks_file:
        chunks_file = "output/chunks_recursive.txt"

    if not Path(chunks_file).exists():
        print(f"❌ Not found: {chunks_file}")
        return

    print(f"\nLoading: {chunks_file}")
    chunks = load_chunks_from_file(chunks_file)

    if not chunks:
        print("❌ No chunks loaded!")
        return

    print(f"✅ Loaded {len(chunks)} chunks")
    print()

    # Choose model
    print("CHOOSE MODEL")
    print("=" * 80)
    print("  1. BGE-M3 (1024 dim) - Best accuracy")
    print("  2. GTE-Viet (768 dim) - Vietnamese optimized")
    print("  3. MiniLM (384 dim) - Fastest")

    choice = input("\nChoice (1-3, default=1): ").strip() or "1"
    print()

    # Setup
    device = "cuda" if has_gpu else "cpu"

    if has_gpu:
        batch_size_map = {"1": 32, "2": 48, "3": 64}  # Giảm xuống cho GPU 4GB
    else:
        batch_size_map = {"1": 32, "2": 32, "3": 64}

    batch_size = batch_size_map.get(choice, 256 if has_gpu else 32)

    # Load embedder
    print("LOADING MODEL")
    print("=" * 80)

    if choice == "1":
        embedder = SingleEmbedder(
            model_name="BAAI/bge-m3",
            device=device,
            batch_size=batch_size
        )
        strategy_name = "bge-m3"

    elif choice == "2":
        embedder = VietnameseEmbedder(
            model_key="gte-viet",
            device=device,
            batch_size=batch_size
        )
        strategy_name = "gte-viet"

    elif choice == "3":
        embedder = SingleEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device=device,
            batch_size=batch_size
        )
        strategy_name = "minilm"

    print()
    print(f"Device: {device.upper()}")
    print(f"Batch size: {batch_size}")
    print()

    # Embed
    print("=" * 80)
    print("EMBEDDING")
    print("=" * 80)
    print(f"Processing {len(chunks)} chunks...")

    if has_gpu:
        print("🔥 Using GPU acceleration!")
        print(f"   Expected time: ~{len(chunks) * 0.05 / 60:.1f} minutes")
    else:
        print("⚠️  Using CPU (slower)")
        print(f"   Expected time: ~{len(chunks) * 0.2 / 60:.1f} minutes")

    print()

    start_time = time()
    embeddings = embedder.encode_chunks(chunks, text_key='text')
    elapsed = time() - start_time

    print()
    print("=" * 80)
    print("✅ EMBEDDING COMPLETED!")
    print("=" * 80)
    print(f"Time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    print(f"Speed: {len(chunks) / elapsed:.1f} chunks/second")
    print(f"Shape: {embeddings.shape}")
    print(f"Memory: {embeddings.nbytes / 1024 / 1024:.2f} MB")

    if has_gpu:
        cpu_time = len(chunks) * 0.2
        speedup = cpu_time / elapsed
        print(f"\n🚀 GPU Speedup: {speedup:.1f}x faster than CPU!")

    print()

    # Save
    print("=" * 80)
    print("SAVING")
    print("=" * 80)

    Path("output/embeddings").mkdir(exist_ok=True, parents=True)

    emb_file = f"output/embeddings/{strategy_name}_embeddings.npy"
    embedder.save_embeddings(embeddings, emb_file)

    chunks_meta = [
        {
            'chunk_id': i,
            'page_num': chunk.get('page_num'),
            'chunk_size': chunk.get('chunk_size', len(chunk['text'])),
            'chunk_type': chunk.get('chunk_type'),
            'text_preview': chunk['text'][:100]
        }
        for i, chunk in enumerate(chunks)
    ]

    meta_file = f"output/embeddings/{strategy_name}_metadata.json"
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump({
            'num_chunks': len(chunks),
            'embedding_dim': int(embeddings.shape[1]),
            'strategy': strategy_name,
            'device': device,
            'batch_size': batch_size,
            'time_seconds': elapsed,
            'chunks': chunks_meta
        }, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved metadata: {meta_file}")

    chunks_file_out = f"output/embeddings/{strategy_name}_chunks.json"
    with open(chunks_file_out, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved chunks: {chunks_file_out}")

    print()
    print("=" * 80)
    print("✅ SUCCESS!")
    print("=" * 80)
    print()
    print("📁 Output files:")
    print(f"   - {emb_file}")
    print(f"   - {meta_file}")
    print(f"   - {chunks_file_out}")
    print()

    if has_gpu:
        print("💡 Performance:")
        print(f"   - Speed: {len(chunks) / elapsed:.1f} chunks/sec")
        print(f"   - Batch size: {batch_size}")
        print()

    print("🔜 Next: python test_retrieval.py")
    print()


if __name__ == "__main__":
    main()