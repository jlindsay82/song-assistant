import chromadb

# Persistent mode — data survives between runs
chroma_client = chromadb.PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection(name="songwriting_guide")