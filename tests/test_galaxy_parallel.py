import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from leonardo_demos.base import RunContext
from leonardo_demos.demos.galaxy_collision import G, GalaxyCollisionDemo, MW_M31
from leonardo_demos.demos.galaxy_collision_3d import GalaxyCollision3DDemo


class GalaxyParallelTests(unittest.TestCase):
    def context(self, root, workers, method="default"):
        with patch.dict(os.environ, {"LEONARDO_DEMO_CPU_WORKERS": str(workers)}):
            return RunContext(Path(root), "galaxy_collision", "local", 2, {}, "numpy",
                              method=method)

    def test_context_parallel_slices_respects_cpu_allocation(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx=self.context(temp,4)
            values=np.zeros(64,dtype=np.int32)
            ctx.parallel_slices(len(values),lambda part: values.__setitem__(part,1),min_items=4)
            self.assertTrue(np.all(values==1))
            self.assertEqual(ctx.cpu_workers,4)
            ctx.finish()
            meta=json.loads((Path(temp)/"meta.json").read_text())
            self.assertEqual(meta["resources"]["cpu_workers"],4)

    def test_parallel_leapfrog_matches_serial_result(self):
        with tempfile.TemporaryDirectory() as serial_dir, tempfile.TemporaryDirectory() as parallel_dir:
            serial_ctx=self.context(serial_dir,1)
            parallel_ctx=self.context(parallel_dir,4)
            serial=GalaxyCollisionDemo(serial_ctx,{})
            parallel=GalaxyCollisionDemo(parallel_ctx,{})
            state=serial.setup(40_000,1,.55,.75,18)
            first=[np.array(value,copy=True) if hasattr(value,"shape") else value for value in state]
            second=[np.array(value,copy=True) if hasattr(value,"shape") else value for value in state]
            got_serial=serial.step(*first[:8],.002,2)
            got_parallel=parallel.step(*second[:8],.002,2)
            for left,right in zip(got_serial,got_parallel):
                np.testing.assert_allclose(left,right,rtol=2e-6,atol=2e-6)
            serial_ctx.finish(); parallel_ctx.finish()

    def test_all_solver_choices_produce_finite_states(self):
        for method in GalaxyCollisionDemo.methods:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp:
                ctx=self.context(temp,1,method)
                demo=GalaxyCollisionDemo(ctx,{})
                state=demo.setup(80,1,.55,.75,18)
                result=demo.step(*state[:8],.001,2)
                for value in result:
                    self.assertTrue(np.isfinite(value).all())
                self.assertEqual(result[0].shape,(80,2))
                ctx.finish()

    def test_murb_solver_matches_reference_constant_acceleration_update(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx=self.context(temp,1,"murb_kinematic")
            demo=GalaxyCollisionDemo(ctx,{})
            state=demo.setup(40,1,.55,.75,18)
            p,v,c1,c2,vc1,vc2,m1,m2=state[:8]
            p0,v0,c10,c20,vc10,vc20=[np.array(x,copy=True) for x in (p,v,c1,c2,vc1,vc2)]
            acceleration=demo.particle_accel(p0,c10,c20,m1,m2)
            centre_a1,centre_a2=demo.centre_accel(c10,c20,m1,m2)
            dt=.001
            result=demo.step(p,v,c1,c2,vc1,vc2,m1,m2,dt,1)
            np.testing.assert_allclose(result[0],p0+dt*v0+.5*dt*dt*acceleration,
                                       rtol=2e-6,atol=2e-6)
            np.testing.assert_allclose(result[1],v0+dt*acceleration,
                                       rtol=2e-6,atol=2e-6)
            np.testing.assert_allclose(result[2],c10+dt*vc10+.5*dt*dt*centre_a1,
                                       rtol=2e-6,atol=2e-6)
            np.testing.assert_allclose(result[3],c20+dt*vc20+.5*dt*dt*centre_a2,
                                       rtol=2e-6,atol=2e-6)
            ctx.finish()

    def test_uncertainty_sweep_builds_an_approaching_consistent_state(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx=self.context(temp,1)
            demo=GalaxyCollisionDemo(ctx,{})
            p,v,c1,c2,vc1,vc2,*_=demo.setup(
                200,1,.55,.75,18,transverse_velocity=42.0)
            relative_velocity=vc2-vc1
            np.testing.assert_allclose(relative_velocity,
                                       [MW_M31["v_radial"],42.0],rtol=1e-6)
            self.assertLess(float(np.dot(c2-c1,relative_velocity)),0.0)
            # Both tracer discs inherit their own centre's bulk velocity.
            self.assertLess(abs(float(np.mean(v[:100,0])-vc1[0])),20.0)
            self.assertLess(abs(float(np.mean(v[100:,0])-vc2[0])),20.0)
            ctx.finish()

    def test_short_run_reveal_integrates_exactly_one_simulation_span(self):
        """The minimum reveal frame count must not multiply physical time."""
        with tempfile.TemporaryDirectory() as temp:
            ctx=self.context(temp,2)
            demo=GalaxyCollisionDemo(ctx,{"particles":200,"substeps":2,"span_gyr":7.5})
            demo.run()
            reveal=np.asarray(Image.open(Path(temp)/"reveal.jpg"))
            bright=np.max(reveal[86:],axis=2)>45
            self.assertGreater(int(bright.sum()),10_000)

    def test_full_3d_force_matches_direct_softened_equation(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx=RunContext(Path(temp),"galaxy_collision_3d","local",2,{},"numpy",
                           method="leapfrog")
            demo=GalaxyCollision3DDemo(ctx,{"force_tile":2})
            positions=np.array(((-2.,0.,.5),(1.,-1.,0.),(.2,3.,-2.)),dtype=np.float32)
            masses=np.array((4e9,2e9,1e9),dtype=np.float32)
            softening=1.7
            got=demo.acceleration(positions,masses,softening)
            expected=np.zeros_like(positions)
            for i in range(len(positions)):
                for j in range(len(positions)):
                    delta=positions[j]-positions[i]
                    radius2=float(delta@delta+softening**2)
                    expected[i]+=G*masses[j]*delta/(radius2*math.sqrt(radius2))
            np.testing.assert_allclose(got,expected,rtol=2e-6,atol=2e-6)
            # Internal pair forces conserve total momentum (up to float error).
            residual=np.sum(got*masses[:,None],axis=0)
            scale=np.sum(np.abs(got*masses[:,None]))
            self.assertLess(float(np.linalg.norm(residual)),float(scale)*2e-6)
            ctx.finish()


if __name__ == "__main__":
    unittest.main()
