# tools/seed_vectorstore.py
import os
import shutil
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer

DEF_PERSIST = os.getenv("PERSIST_DIRECTORY", "/tmp/chroma")
DEF_NAME = os.getenv("VECTORSTORE_NAME", "products")
DEF_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# 10 przykładowych pozycji (offline demo)
DEMO_PRODUCTS: List[Dict] = [
    {
        "title": "Apple MacBook Air 13 (M2, 8GB/256GB)",
        "description": "Ultrabook 13.6\" Liquid Retina, chip Apple M2, 8GB RAM, 256GB SSD, 2x Thunderbolt/USB 4, Wi-Fi 6. Waga ~1.24 kg.",
        "url": "https://example.com/macbook-air-m2",
        "price": 1099.0,
    },
    {
        "title": "Dell XPS 13 (i7-1360P, 16GB/512GB)",
        "description": "13.4\" 1920x1200, Intel Core i7-1360P, 16GB LPDDR5, 512GB NVMe, 2x TB4, Wi-Fi 6E, ~1.17 kg.",
        "url": "https://example.com/dell-xps-13",
        "price": 1299.0,
    },
    {
        "title": "Lenovo ThinkPad T14 Gen 4 (Ryzen 7 PRO, 16GB/512GB)",
        "description": "14\" IPS, AMD Ryzen 7 PRO, 16GB, 512GB SSD, RJ-45 via adapter, USB-C/USB-A/HDMI, MIL-STD-810H.",
        "url": "https://example.com/thinkpad-t14",
        "price": 1199.0,
    },
    {
        "title": "ASUS ROG Strix G15 (RTX 4060)",
        "description": "15.6\" 144Hz, Ryzen 7 6800H, GeForce RTX 4060 8GB, 16GB RAM, 1TB SSD NVMe, Wi-Fi 6.",
        "url": "https://example.com/asus-rog-strix-g15",
        "price": 1399.0,
    },
    {
        "title": "LG 27UL850 27\" 4K IPS USB-C",
        "description": "Monitor 3840x2160, IPS, 60Hz, HDR10, USB-C (power delivery), sRGB ~99%, pivot.",
        "url": "https://example.com/lg-27ul850",
        "price": 369.0,
    },
    {
        "title": "Dell UltraSharp U2720Q 27\" 4K",
        "description": "IPS 4K, 99% sRGB/95% DCI-P3, USB-C 90W, hub USB, precyzyjna kalibracja fabryczna.",
        "url": "https://example.com/dell-u2720q",
        "price": 499.0,
    },
    {
        "title": "Logitech MX Master 3S",
        "description": "Mysz bezprzewodowa z elektromagnetycznym kółkiem MagSpeed, cicha, Bluetooth/Logi Bolt, do 3 urządzeń.",
        "url": "https://example.com/logitech-mx-master-3s",
        "price": 89.0,
    },
    {
        "title": "Keychron K2 Pro (Gateron G Pro Brown)",
        "description": "Klawiatura mechaniczna 75%, hot-swap, Bluetooth 5.1, PBT keycaps (opcjonalnie), Mac/Win.",
        "url": "https://example.com/keychron-k2-pro",
        "price": 99.0,
    },
    {
        "title": "Sony WH-1000XM5",
        "description": "Słuchawki z ANC, multipoint BT, do 30h pracy, szybkie ładowanie USB-C, świetny mikrofon.",
        "url": "https://example.com/sony-wh-1000xm5",
        "price": 349.0,
    },
    {
        "title": "Anker 737 Power Bank (PowerCore 24K)",
        "description": "Powerbank 24 000 mAh, 140W PD 3.1, wyświetlacz, 2×USB-C + 1×USB-A, ładowanie dwukierunkowe.",
        "url": "https://example.com/anker-737",
        "price": 159.0,
    },
]

def _safe_reset_dir(path: str):
    try:
        shutil.rmtree(path)
    except Exception:
        pass
    os.makedirs(path, exist_ok=True)

def seed_demo_products(path: str, name: str, extra_docs: List[Dict] | None = None) -> int:
    """Seeduje kolekcję 10 (lub więcej) produktów demo – zawsze działa offline."""
    docs = list(DEMO_PRODUCTS if extra_docs is None else extra_docs)  # kopia
    encoder = SentenceTransformer(DEF_MODEL)
    texts = [f"{d['title']}. {d['description']}".strip() for d in docs]
    embeddings = encoder.encode(texts, normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=path)
    col = client.get_or_create_collection(name=name)

    ids = [f"demo-{i}" for i in range(len(docs))]
    metadatas = [{"price": d["price"], "url": d["url"], "title": d["title"]} for d in docs]
    col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"[seed_vectorstore] Seeded {len(docs)} demo items into Chroma '{name}' at '{path}'")
    return len(docs)

def seed_vectorstore(
    limit_per_feed: int = 3,
    path: str | None = None,
    name: str | None = None,
    reset: bool = False,
) -> None:
    path = path or os.getenv("VECTORSTORE_PATH", DEF_PERSIST)
    name = name or os.getenv("VECTORSTORE_NAME", DEF_NAME)

    os.makedirs(path, exist_ok=True)

    # 1) Spróbuj usunąć starą kolekcję, jeśli reset=True
    client = chromadb.PersistentClient(path=path)
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    # 2) Najpierw spróbuj zasilić z RSS (jak w Twojej oryginalnej wersji)
    try:
        from agents.deals import ScrapedDeal
        scraped: List[ScrapedDeal] = ScrapedDeal.fetch(
            show_progress=False, limit_per_feed=limit_per_feed, fetch_page=False
        )
        docs = [{
            "title": s.title,
            "description": f"{s.title}. {s.summary} {s.details} {s.features}".strip(),
            "url": s.url,
            "price": 0.0
        } for s in scraped][: 5 * limit_per_feed]
    except Exception:
        docs = []

    # 3) Jeśli RSS zwróci mało/zero, zasiej demo (minimum 10)
    if not docs or len(docs) < 10:
        seed_demo_products(path, name)  # 10 itemów
        return

    # 4) Inaczej – zasiej to, co przyszło z RSS
    encoder = SentenceTransformer(DEF_MODEL)
    texts = [d["description"] for d in docs]
    embeddings = encoder.encode(texts, normalize_embeddings=True).tolist()

    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    col = client.get_or_create_collection(name=name)
    ids = [f"deal-{i}" for i in range(len(docs))]
    metadatas = [{"price": d["price"], "url": d["url"], "title": d["title"]} for d in docs]
    col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"[seed_vectorstore] Seeded {len(docs)} items into Chroma '{name}' at '{path}'")

if __name__ == "__main__":
    # Force reset + seed (z RSS lub demo)
    seed_vectorstore(reset=True)
