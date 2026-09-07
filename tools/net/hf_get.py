"""Download an HF repo through the DNS-pinned socket layer."""
import sys
import dns8888  # noqa: F401  (patches socket.getaddrinfo)
from huggingface_hub import snapshot_download

repo, local_dir = sys.argv[1], sys.argv[2]
allow = sys.argv[3:] or None
p = snapshot_download(repo_id=repo, local_dir=local_dir,
                      allow_patterns=allow, max_workers=8)
print("DONE ->", p)
