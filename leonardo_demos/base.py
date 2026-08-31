from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
import json, time, traceback
from typing import Any, Dict
from .backend import choose_backend

@dataclass
class RunContext:
    run_dir: Path
    demo: str
    profile: str
    frames: int
    params: Dict[str, Any]
    backend_requested: str = "auto"
    xp: Any = field(init=False, repr=False)
    backend_name: str = field(init=False)
    _frame_pool: ThreadPoolExecutor | None = field(init=False, default=None, repr=False)
    _frame_futures: list[Future] = field(init=False, default_factory=list, repr=False)
    started: float = field(default_factory=time.time)

    def __post_init__(self):
        # On Windows the CUDA runtimes bundled by PyTorch and CuPy must be
        # loaded in a consistent order.  The neural wall uses PyTorch for its
        # batched training; importing it before the generic CuPy probe prevents
        # a first CUDA call from hanging when the GPU option is selected.
        if self.demo == "neural_wall":
            try:
                import torch  # noqa: F401
            except Exception:
                pass
        self.run_dir.mkdir(parents=True,exist_ok=True)
        (self.run_dir/'frames').mkdir(exist_ok=True)
        self.xp,self.backend_name=choose_backend(self.backend_requested)
        if self.backend_requested.lower() in {"hybrid", "cpu+gpu", "cpu_gpu"}:
            # GPU simulation and Pillow's JPEG encoder use different hardware.
            # Keep the queue short so the CPU overlaps the next CUDA update
            # without retaining a whole run's worth of images in memory.
            self._frame_pool=ThreadPoolExecutor(max_workers=2,
                                                 thread_name_prefix="frame-encode")
        self.write_meta({"status":"starting","demo":self.demo,"profile":self.profile,"frames":self.frames,"params":self.params,"backend":self.backend_name,"created":self.started})

    def write_meta(self, update):
        p=self.run_dir/'meta.json'
        base={}
        if p.exists():
            try: base=json.loads(p.read_text())
            except Exception: pass
        base.update(update)
        p.write_text(json.dumps(base,indent=2))

    def frame_path(self,i): return self.run_dir/'frames'/f'frame_{i:04d}.jpg'
    def save_frame(self, image, path: Path):
        """Write a frame, overlapping CPU encoding with CUDA work in hybrid mode."""
        if self._frame_pool is None:
            from .render import save_frame
            save_frame(image,path)
            return
        copy=image.copy()
        self._frame_futures.append(self._frame_pool.submit(self._save_atomic,copy,Path(path)))
        if len(self._frame_futures)>2:
            self._frame_futures.pop(0).result()

    @staticmethod
    def _save_atomic(image,path: Path):
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(f'.{path.stem}.{time.time_ns()}.tmp{path.suffix}')
        image.save(temporary,quality=92)
        temporary.replace(path)

    def flush_frames(self):
        while self._frame_futures:
            self._frame_futures.pop(0).result()
        if self._frame_pool is not None:
            self._frame_pool.shutdown(wait=True)
            self._frame_pool=None
    def write_status(self, frame, message=""):
        self.write_meta({"status":"running","frame":frame,"message":message,"elapsed":time.time()-self.started})
    def finish(self, reveal=None):
        self.flush_frames()
        self.write_meta({"status":"complete","frame":self.frames-1,"elapsed":time.time()-self.started,"reveal":str(reveal.name) if reveal else None})
    def fail(self, exc):
        try: self.flush_frames()
        except Exception: pass
        self.write_meta({"status":"failed","error":str(exc),"traceback":traceback.format_exc()})

class Demo:
    id="base"
    title="Demo"
    def __init__(self, ctx, settings): self.ctx=ctx; self.settings=settings
    def run(self): raise NotImplementedError
