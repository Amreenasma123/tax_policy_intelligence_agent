"""Read-only inspection of the local TaxGuard Chroma collection."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Inspect TaxGuard's persistent Chroma collection")
    parser.add_argument("--path", default="data/chroma", help="Chroma persistence directory")
    parser.add_argument("--collection", default="taxguard_chunks")
    parser.add_argument("--limit", type=int, default=5, help="Number of sample records to display")
    parser.add_argument("--query", default=None, help="Optional semantic query to test stored vectors")
    args = parser.parse_args()
    try:
        import chromadb
    except ImportError:
        print("Chroma isn't installed. Run: pip install -r requirements-ml.txt")
        return 2
    path = str((Path.cwd() / args.path).resolve())
    print(f"Persistence path: {path}")
    client = chromadb.PersistentClient(path=path)
    names = [c.name for c in client.list_collections()]
    print(f"Collections: {names or '(none)'}")
    if args.collection not in names:
        print(f"Collection '{args.collection}' not found. Run: python scripts/taxguard.py ingest --dense")
        return 1
    collection = client.get_collection(args.collection)
    print(f"Collection: {collection.name}")
    print(f"Record count: {collection.count()}")
    print(f"Metadata: {collection.metadata}")
    if collection.count() == 0:
        print("Collection exists but contains no records.")
        return 0
    if args.query:
        result = collection.query(query_texts=[args.query], n_results=min(max(1, args.limit), collection.count()), include=["documents", "metadatas", "distances"])
        print("\nSemantic query results (lower cosine distance is generally closer):")
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i, cid in enumerate(ids):
            print(f"\n{i+1}. id={cid} distance={dists[i] if i < len(dists) else 'n/a'}")
            print(f"   metadata={metas[i] if i < len(metas) else {}}")
            print(f"   text={(docs[i] if i < len(docs) else '')[:350].replace(chr(10), ' ')}")
    else:
        result = collection.get(limit=min(max(1, args.limit), collection.count()), include=["documents", "metadatas"])
        print("\nSample stored chunks:")
        for i, cid in enumerate(result.get("ids", [])):
            print(f"\n{i+1}. id={cid}")
            print(f"   metadata={result.get('metadatas', [])[i]}")
            print(f"   text={result.get('documents', [])[i][:350].replace(chr(10), ' ')}")
    print("\nNote: Chroma stores vectors, chunk text and metadata; this script only reads and displays records.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
