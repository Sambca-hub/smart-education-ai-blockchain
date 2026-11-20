import hashlib
import json
import os
from datetime import datetime, timezone


CHAIN_FILE = "blockchain_data.json"


class Blockchain:
    """A local SHA-256 hash-linked record chain (not a distributed blockchain)."""

    def __init__(self, storage_path=CHAIN_FILE):
        self.storage_path = storage_path
        self.chain = []
        self.load_chain()

        if not self.chain:
            self.create_genesis_block()
            self.save_chain()

    @staticmethod
    def calculate_hash(block):
        block_data = {
            "index": block["index"],
            "timestamp": block["timestamp"],
            "data": block["data"],
            "previous_hash": block["previous_hash"],
        }
        encoded = json.dumps(
            block_data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def create_genesis_block(self):
        block = {
            "index": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"message": "Genesis Block"},
            "previous_hash": "0",
        }
        block["hash"] = self.calculate_hash(block)
        self.chain.append(block)

    def add_block(self, data):
        previous = self.chain[-1]
        block = {
            "index": len(self.chain),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
            "previous_hash": previous["hash"],
        }
        block["hash"] = self.calculate_hash(block)
        self.chain.append(block)
        self.save_chain()
        return block

    def verify_chain(self):
        if not self.chain:
            return False, "Blockchain is empty."

        for index, block in enumerate(self.chain):
            if block.get("index") != index:
                return False, f"Block {index} has an invalid index."

            if block.get("hash") != self.calculate_hash(block):
                return False, f"Block {index} has been modified."

            if index == 0:
                if block.get("previous_hash") != "0":
                    return False, "Genesis block has an invalid previous hash."
            else:
                previous = self.chain[index - 1]
                if block.get("previous_hash") != previous.get("hash"):
                    return False, (
                        f"Block {index} has an invalid previous hash."
                    )

        return True, "All blockchain records are valid."

    def save_chain(self):
        directory = os.path.dirname(self.storage_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        temporary_path = self.storage_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(
                self.chain, file, indent=4, ensure_ascii=False
            )
        os.replace(temporary_path, self.storage_path)

    def load_chain(self):
        if not os.path.exists(self.storage_path):
            self.chain = []
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as file:
                loaded = json.load(file)
            self.chain = loaded if isinstance(loaded, list) else []
        except (json.JSONDecodeError, OSError):
            self.chain = []

    def get_chain(self):
        return self.chain
