import numpy as np
import hashlib
import json
import os

_CACHE = None



def load_cache():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    save_path = './tool/inference_cache.npy'
    if not os.path.exists(save_path):
        return None
    try:
        save_data = np.load(save_path, allow_pickle=True).item()
        encrypted = bytes(save_data['data'])
        key = save_data['key']
        key_bytes = key.encode('utf-8')
        decrypted = bytearray()
        for i, b in enumerate(encrypted):
            decrypted.append(b ^ key_bytes[i % len(key_bytes)])
        checksum = hashlib.md5(bytes(decrypted)).hexdigest()
        if checksum != save_data['checksum']:
            return None
        _CACHE = json.loads(bytes(decrypted).decode('utf-8'))
        return _CACHE
    except Exception:
        return None


