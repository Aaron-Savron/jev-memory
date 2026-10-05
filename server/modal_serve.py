"""Optional persistent deployment. Create the authentication secret first."""
import sys
from pathlib import Path
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modal_app import app, image, cache


@app.cls(image=image, gpu="T4", memory=12288, max_containers=1,
         scaledown_window=300, timeout=600, volumes={"/models": cache},
         secrets=[modal.Secret.from_name("jev-memory-server")])
class DecisionService:
    @modal.enter()
    def load(self):
        sys.path.insert(0, "/srv/memory")
        from backend import MemoryBackend
        self.backend = MemoryBackend()
        cache.commit()

    @modal.asgi_app()
    def web(self):
        from app import create_app
        return create_app(lambda: self.backend)
