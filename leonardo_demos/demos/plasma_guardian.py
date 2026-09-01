from __future__ import annotations

"""A deliberately reduced, trainable plasma-feedback-control demonstration.

This is not a tokamak equilibrium or disruption solver.  It is a small,
differentiable state-space environment inspired by the control loop used in
tokamaks: noisy diagnostics -> neural policy -> coil commands -> new plasma
state.  The policy is genuinely optimized through the simulated dynamics.
"""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..backend import torch_device
from ..base import Demo
from ..render import font


INPUT_NAMES = ("radial", "vertical", "radial v", "vertical v", "tearing", "pressure")
OUTPUT_NAMES = ("radial coils", "vertical coils", "shape coils")


class AnalyticSafetyPolicy:
    """Explicit fallback when PyTorch is unavailable; it never claims training."""

    device = "analytical fallback"
    training = False

    def train(self, updates, batch, horizon, drive):
        return 0.0

    @staticmethod
    def actions(state):
        state = np.asarray(state, dtype=np.float32)
        return np.clip(np.stack((
            -1.20 * state[..., 0] - 0.56 * state[..., 2],
            -1.20 * state[..., 1] - 0.56 * state[..., 3],
            1.0 * state[..., 4] + 0.18 * state[..., 5],
        ), axis=-1), -1, 1)

    def evaluate(self, steps, drive, controlled):
        state = np.array([0.34, -0.28, 0.08, -0.06, 0.28, 0.88], dtype=np.float32)
        return rollout_numpy(state, steps, drive, self.actions if controlled else None)

    def weights(self):
        rng = np.random.default_rng(7)
        return (rng.normal(0, .4, (6, 14)), rng.normal(0, .32, (14, 10)), rng.normal(0, .48, (10, 3)))


def transition_numpy(state, action, disturbance, step, drive):
    """One step of the reduced unstable plasma state space.

    State is radial/vertical position and velocity, a tearing-risk proxy and a
    pressure proxy.  The positive position feedback and growing tearing proxy
    create an unstable open loop.  Three aggregate coil banks counter it.
    """
    x, y, vx, vy, mode, pressure = state
    ar, av, ashape = action
    phase = .21 * step
    kick_r = drive * (.038 * math.sin(phase * 1.7) + .020 * math.sin(phase * 3.1))
    kick_v = drive * (.034 * math.cos(phase * 1.3) - .018 * math.sin(phase * 2.4))
    vx = vx + .075 * (0.78 * x - .48 * vx + 1.52 * ar + kick_r)
    vy = vy + .075 * (0.82 * y - .44 * vy + 1.52 * av + kick_v)
    x = x + .075 * vx
    y = y + .075 * vy
    wall = x * x + y * y
    pressure = np.clip(pressure + .075 * (.09 * drive - .075 * (ashape + 1) / 2 - .045 * wall), .45, 1.35)
    mode = np.clip(mode + .075 * (.22 * drive + 1.22 * wall + .15 * (vx * vx + vy * vy) - .94 * (ashape + 1) / 2), 0, 1.6)
    return np.array([x, y, vx, vy, mode, pressure], dtype=np.float32)


def rollout_numpy(start, steps, drive, policy=None):
    state = np.asarray(start, dtype=np.float32).copy()
    states, actions = [], []
    for k in range(steps):
        action = np.zeros(3, dtype=np.float32) if policy is None else np.asarray(policy(state), dtype=np.float32)
        action = np.clip(action, -1, 1)
        states.append(state.copy()); actions.append(action)
        state = transition_numpy(state, action, np.zeros(2), k, drive)
    return np.asarray(states), np.asarray(actions)


class TorchPolicy:
    """Small neural policy optimized through batches of virtual plasma shots."""

    training = True

    def __init__(self, requested):
        import torch

        self.torch = torch
        self.device = torch_device(requested)
        torch.manual_seed(23)
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(6, 14), torch.nn.Tanh(),
            torch.nn.Linear(14, 10), torch.nn.Tanh(),
            torch.nn.Linear(10, 3), torch.nn.Tanh(),
        ).to(self.device)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=0.010)
        # Trigger the backend's first matmul while construction is still inside
        # make_trainer.  A CUDA library can report a device then fail only when
        # cuBLAS initialises; Auto mode can safely retry on CPU in that case.
        with torch.no_grad():
            self.policy(torch.zeros((1, 6), device=self.device)).sum().item()
        self.last_loss = 0.0

    def _transition(self, state, step, drive):
        torch = self.torch
        action = self.policy(state)
        x, y, vx, vy, mode, pressure = state.unbind(-1)
        ar, av, ashape = action.unbind(-1)
        phase = .21 * step
        kick_r = drive * (.038 * math.sin(phase * 1.7) + .020 * math.sin(phase * 3.1))
        kick_v = drive * (.034 * math.cos(phase * 1.3) - .018 * math.sin(phase * 2.4))
        vx = vx + .075 * (.78 * x - .48 * vx + 1.52 * ar + kick_r)
        vy = vy + .075 * (.82 * y - .44 * vy + 1.52 * av + kick_v)
        x = x + .075 * vx
        y = y + .075 * vy
        wall = x.square() + y.square()
        pressure = torch.clamp(pressure + .075 * (.09 * drive - .075 * (ashape + 1) / 2 - .045 * wall), .45, 1.35)
        mode = torch.clamp(mode + .075 * (.22 * drive + 1.22 * wall + .15 * (vx.square() + vy.square()) - .94 * (ashape + 1) / 2), 0, 1.6)
        return torch.stack((x, y, vx, vy, mode, pressure), -1), action

    def train(self, updates, batch, horizon, drive):
        torch = self.torch
        losses = []
        for _ in range(max(1, int(updates))):
            state = torch.empty((batch, 6), device=self.device).uniform_(-1, 1)
            state[:, :2] *= .46; state[:, 2:4] *= .14
            state[:, 4] = torch.empty(batch, device=self.device).uniform_(.08, .52)
            state[:, 5] = torch.empty(batch, device=self.device).uniform_(.70, 1.18)
            loss = torch.zeros((), device=self.device)
            for step in range(horizon):
                state, action = self._transition(state, step, drive)
                wall = state[:, 0].square() + state[:, 1].square()
                loss = loss + (3.2 * wall + 1.9 * state[:, 4].square() + .22 * state[:, 2:4].square().sum(-1) + .015 * action.square().sum(-1)).mean()
            loss = loss / horizon
            self.optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optim.step()
            losses.append(float(loss.detach().cpu()))
        self.last_loss = float(np.mean(losses))
        return self.last_loss

    def evaluate(self, steps, drive, controlled):
        torch = self.torch
        state = torch.tensor([[.34, -.28, .08, -.06, .28, .88]], device=self.device)
        states, actions = [], []
        with torch.no_grad():
            for k in range(steps):
                if controlled:
                    state, action = self._transition(state, k, drive)
                else:
                    value = state[0].detach().cpu().numpy()
                    nxt = transition_numpy(value, np.zeros(3, dtype=np.float32), np.zeros(2), k, drive)
                    action = torch.zeros((1, 3), device=self.device)
                    state = torch.tensor(nxt[None], device=self.device)
                states.append(state[0].detach().cpu().numpy()); actions.append(action[0].detach().cpu().numpy())
        return np.asarray(states), np.asarray(actions)

    def weights(self):
        layers = [m.weight.detach().cpu().numpy().T for m in self.policy if hasattr(m, "weight")]
        return tuple(layers)


class PlasmaGuardianDemo(Demo):
    id = "plasma_guardian"
    title = "AI Plasma Guardian"
    backend_kind = "torch"
    timing_methods = {"train":"simulation", "render_frame":"render"}

    def make_trainer(self):
        requested = getattr(self.ctx, "backend_requested", "auto")
        try:
            trainer = TorchPolicy(requested)
            self.ctx.set_backend_name(f"torch·{trainer.device}")
            return trainer
        except Exception as exc:
            if requested.lower() == "auto":
                try:
                    trainer = TorchPolicy("cpu")
                    self.ctx.set_backend_name("torch·cpu (CUDA unavailable at runtime)")
                    return trainer
                except Exception:
                    pass
            requested = requested.lower()
            if requested in {"gpu", "cuda", "cupy", "hybrid", "cpu+gpu", "cpu_gpu"}:
                raise RuntimeError(f"GPU requested for plasma_guardian but PyTorch could not start: {exc}") from exc
            self.ctx.set_backend_name("analytical fallback — PyTorch unavailable")
            return AnalyticSafetyPolicy()

    def train(self, trainer, updates, drive):
        return trainer.train(updates, int(self.settings.get("batch", 128)), int(self.settings.get("horizon", 28)), drive)

    @staticmethod
    def _risk(state):
        return float(np.clip(.35 * (state[0] ** 2 + state[1] ** 2) + .70 * state[4] ** 2 + .11 * state[2:4].dot(state[2:4]), 0, 1.0))

    @staticmethod
    def _ellipse_points(cx, cy, rx, ry, wobble, count=96):
        theta = np.linspace(0, math.tau, count)
        r = 1 + wobble * np.sin(5 * theta + 1.7) + wobble * .45 * np.sin(9 * theta - .4)
        return [(cx + rx * r0 * math.cos(t), cy + ry * r0 * math.sin(t)) for t, r0 in zip(theta, r)]

    def draw_plasma(self, draw, state, action, baseline_state, phase):
        cx, cy = 640, 370; vessel_r = 250
        # Vessel and the three independently commanded aggregate coil banks.
        draw.ellipse((cx-vessel_r, cy-vessel_r, cx+vessel_r, cy+vessel_r), outline=(79, 133, 183, 210), width=4)
        draw.ellipse((cx-vessel_r+17, cy-vessel_r+17, cx+vessel_r-17, cy+vessel_r-17), outline=(24, 63, 105, 180), width=2)
        coil_specs = [((cx-258,cy-85,cx-238,cy+85), action[0], (80, 225, 255)),
                      ((cx-82,cy-258,cx+82,cy-238), action[1], (255, 166, 89)),
                      ((cx-82,cy+238,cx+82,cy+258), action[2], (198, 126, 255))]
        for box, command, colour in coil_specs:
            level = .25 + .75 * abs(float(command))
            draw.rounded_rectangle(box, radius=7, outline=(*colour, int(110 + 140 * level)), width=5)
            centre = ((box[0]+box[2]) / 2, (box[1]+box[3]) / 2)
            for r in (15, 27, 40):
                draw.ellipse((centre[0]-r,centre[1]-r,centre[0]+r,centre[1]+r), outline=(*colour, int(26 * level)), width=2)
        # The ghost is an uncontrolled reference trajectory, not a second plasma.
        # Keep the failed reference inside the vessel frame so its impending
        # wall contact remains readable even after the reduced model diverges.
        bx = cx + float(np.clip(baseline_state[0], -.92, .92)) * 225
        by = cy + float(np.clip(baseline_state[1], -.92, .92)) * 225
        bw = 15 + 48 * min(1, float(baseline_state[4]))
        draw.line(self._ellipse_points(bx, by, 72+bw, 54+bw*.7, .018+baseline_state[4]*.025), fill=(255,80,91,120), width=3, joint="curve")
        px = cx + state[0] * 225; py = cy + state[1] * 225
        mode = float(np.clip(state[4], 0, 1)); pressure=float(np.clip(state[5], .45, 1.35))
        rx = 118 * (1 + .10*(pressure-.9)); ry = 84 * (1 - .08*(pressure-.9))
        glow = Image.new("RGBA", (1280,720), (0,0,0,0)); gd = ImageDraw.Draw(glow, "RGBA")
        for expansion, alpha in ((30,28),(18,44),(8,80)):
            gd.polygon(self._ellipse_points(px,py,rx+expansion,ry+expansion, .025+mode*.10), fill=(53,199,255,alpha))
        glow = glow.filter(ImageFilter.GaussianBlur(12))
        draw._image.alpha_composite(glow)
        draw.polygon(self._ellipse_points(px, py, rx, ry, .024 + .13*mode), fill=(77, 220, 255, 220), outline=(211,251,255,245))
        for fraction in (.35,.62,.82):
            draw.line(self._ellipse_points(px, py, rx*fraction, ry*fraction, .012+mode*.08), fill=(205,250,255,80), width=1)
        # A smooth amber island is part of the simulated state: its size is the
        # tearing-risk proxy.  Drawing it after the plasma avoids the clipped,
        # star-like artefact produced by the old layer order.
        island_angle = phase * .19
        island_x = px + (rx*.60) * math.cos(island_angle)
        island_y = py + (ry*.42) * math.sin(island_angle)
        island = 12 + 58 * mode
        draw.ellipse((island_x-island,island_y-island*.62,island_x+island,island_y+island*.62), fill=(255,139,76,int(60+150*mode)), outline=(255,221,153,220), width=2)
        draw.ellipse((px-7,py-7,px+7,py+7),fill=(250,255,255,250))

    @staticmethod
    def _node_positions(x, top, bottom, n):
        return [(x, top + (bottom-top)*(i+.5)/n) for i in range(n)]

    def draw_network(self, weights, action, loss, training, progress):
        image = Image.new("RGBA", (530, 520), (4, 12, 27, 235))
        d = ImageDraw.Draw(image, "RGBA")
        d.rounded_rectangle((0,0,529,519),radius=18,outline=(65,155,213,190),width=2)
        d.text((22,18), "NEURAL FEEDBACK POLICY", font=font(17, True), fill=(231,246,255,255))
        subtitle = "training through virtual plasma shots" if training else "analytical fallback — not learning"
        d.text((22,45), subtitle, font=font(11), fill=(142,191,224,255))
        xs=(54,190,332,472); layers=(self._node_positions(xs[0],94,402,6), self._node_positions(xs[1],118,378,7), self._node_positions(xs[2],132,364,6), self._node_positions(xs[3],178,318,3))
        compact=(weights[0][:, :7], weights[1][:7, :6], weights[2][:6, :])
        for stage, matrix in enumerate(compact):
            scale=max(.02,float(np.percentile(np.abs(matrix),90)))
            for a, start in enumerate(layers[stage]):
                for b, end in enumerate(layers[stage+1]):
                    value=float(matrix[a,b]); strength=min(1,abs(value)/scale)
                    colour=(78,226,255,int(18+150*strength)) if value>=0 else (250,92,177,int(18+150*strength))
                    d.line((*start,*end),fill=colour,width=1+int(2*strength))
        for layer, nodes in enumerate(layers):
            for idx, (x,y) in enumerate(nodes):
                if layer==3:
                    magnitude=abs(float(action[idx])); colour=((83,232,255) if idx==0 else (255,170,88) if idx==1 else (206,126,255))
                    r=9+5*magnitude
                else:
                    value=.55+.45*math.sin(progress*12 + idx*1.8 + layer)
                    colour=(95,207,255); r=6+2*value
                d.ellipse((x-r,y-r,x+r,y+r),fill=(*colour,235),outline=(230,250,255,245),width=1)
            if layer==0:
                for idx,(x,y) in enumerate(nodes): d.text((x-43,y-5),INPUT_NAMES[idx],font=font(9),fill=(156,196,224,240))
            if layer==3:
                for idx,(x,y) in enumerate(nodes): d.text((x+16,y-5),OUTPUT_NAMES[idx],font=font(9),fill=(196,222,245,245))
        d.text((22,470), f"policy loss  {loss:.4f}",font=font(13,True),fill=(127,239,255,255))
        d.text((305,470), "cyan + / pink − weight",font=font(10),fill=(170,197,224,220))
        return image.convert("RGB")

    def render_frame(self, state, action, baseline, weights, loss, training, step, total, drive):
        image = Image.new("RGBA", (1280,720), (2,8,19,255))
        d=ImageDraw.Draw(image,"RGBA"); d._image=image
        # Main stream is only the controlled vessel state. Sensor bars, loss,
        # labels and policy topology are supplied by independently toggleable
        # browser layers.
        for x in range(80, 1240, 40): d.line((x,42,x,698),fill=(30,71,112,25),width=1)
        for y in range(50,700,40): d.line((40,y,1240,y),fill=(30,71,112,25),width=1)
        self.draw_plasma(d, state, action, baseline, step)
        return image.convert("RGB")

    def run(self):
        drive=float(self.ctx.params.get("instability", 1.0))
        trainer=self.make_trainer(); total_updates=max(1,int(self.settings.get("train_updates", 120)))
        rollout_steps=int(self.settings.get("display_steps", 96)); done=0; best_loss=0.0
        (self.ctx.run_dir / "overlays" / "network").mkdir(parents=True, exist_ok=True)
        for i in range(self.ctx.frames):
            target=int(round(total_updates*(i+1)/self.ctx.frames))
            best_loss=self.train(trainer,max(1,target-done),drive); done=target
            controlled, actions=trainer.evaluate(rollout_steps,drive,True)
            baseline,_=trainer.evaluate(rollout_steps,drive,False)
            phase=min(rollout_steps-1,int((i+1)/self.ctx.frames*rollout_steps))
            state=controlled[phase]; action=actions[phase]; reference=baseline[phase]
            weights=trainer.weights()
            image=self.render_frame(state,action,reference,weights,best_loss,trainer.training,i,total_updates,drive)
            self.ctx.save_frame(image,self.ctx.frame_path(i))
            graph=self.draw_network(weights,action,best_loss,trainer.training,(i+1)/self.ctx.frames)
            self.ctx.save_frame(graph,(self.ctx.run_dir/"overlays"/"network"/f"frame_{i:04d}.jpg"))
            status="policy optimizing" if trainer.training else "analytical safety fallback"
            self.ctx.write_status(i,status,{
                "training updates":f"{done:,}", "virtual shots / update":f"{int(self.settings.get('batch',128)):,}",
                "policy loss":f"{best_loss:.4f}", "tearing-risk proxy":f"{self._risk(state):.2f}",
                "uncontrolled risk":f"{self._risk(reference):.2f}", "coil command norm":f"{np.linalg.norm(action):.2f}",
                "control model":"neural policy" if trainer.training else "analytical fallback",
            })
        # The last frame is a useful stable replay thumbnail; the model does not
        # fabricate a separate ensemble sweep.
        reveal=self.ctx.frame_path(self.ctx.frames-1)
        self.ctx.finish(reveal)
