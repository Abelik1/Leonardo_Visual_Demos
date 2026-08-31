import base64
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from fastapi import HTTPException

from app import RunReq, save_target_image, start
from leonardo_demos.base import RunContext
from leonardo_demos.demos.neural_wall import NeuralWallDemo
from run_demo import run


class InputValidationTests(unittest.TestCase):
    def test_run_rejects_nonpositive_frame_count(self):
        with self.assertRaisesRegex(ValueError, 'frames must be at least 1'):
            run('reaction_diffusion', frames=0)

    def test_api_rejects_unknown_profile(self):
        with self.assertRaises(HTTPException):
            start('reaction_diffusion', RunReq(profile='invalid'))

    def test_api_rejects_unknown_and_out_of_range_parameters(self):
        with self.assertRaises(HTTPException):
            start('reaction_diffusion', RunReq(params={'unexpected': 1}))
        with self.assertRaises(HTTPException):
            start('reaction_diffusion', RunReq(params={'feed': 1.0}))

    def test_neural_difficulty_changes_the_training_target(self):
        demo = NeuralWallDemo(None, {})
        easy = demo.target(24, kind=1, difficulty=0.5)
        hard = demo.target(24, kind=1, difficulty=2.0)
        self.assertFalse((easy == hard).all())

    def test_canvas_png_becomes_a_neural_target(self):
        image=Image.new('L',(32,32),0)
        for y in range(12,21):
            for x in range(12,21):
                image.putpixel((x,y),255)
        buf=BytesIO(); image.save(buf,format='PNG')
        data='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'target.png'
            save_target_image(data,path)
            target=NeuralWallDemo(None,{}).drawn_target(16,path)
        self.assertGreater(float(target.max()), .5)
        self.assertLess(float(target[0,0].max()), .01)

    def test_neural_targets_are_rgb(self):
        self.assertEqual(NeuralWallDemo(None,{}).target(16,kind=2).shape,(16,16,3))

    def test_neural_hero_renders_live_weight_view(self):
        demo=NeuralWallDemo(None,{})
        state={'width':6,'w1':np.ones((2,6),dtype=np.float32),
               'w2':np.eye(6,dtype=np.float32),'w3':np.ones((6,1),dtype=np.float32)}
        image=demo.hero_image(np.zeros((16,16)),np.ones((16,16)),np.array([.1]),0,
                              [.5,.1],1,'torch·cpu',1.0,20,100,True,state)
        self.assertEqual(image.size,(1280,720))

    def test_neural_demo_renders_network_view(self):
        with tempfile.TemporaryDirectory() as directory:
            context=RunContext(Path(directory),'neural_wall','local',2,{'target':0,'difficulty':1.0},'numpy')
            NeuralWallDemo(context,{'networks':4,'tile':10,'total_steps':4}).run()
            self.assertTrue((Path(directory)/'frames'/'frame_0000.jpg').exists())


if __name__ == '__main__':
    unittest.main()
