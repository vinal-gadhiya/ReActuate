"""Run on Thor: python /opt/smoke_test.py [--kitchen] [--output /outputs]."""

import argparse
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
from OpenGL import GL


def check_mujoco(output):
    model = mujoco.MjModel.from_xml_string('''
    <mujoco><worldbody>
      <light pos="0 0 3"/>
      <geom type="plane" size="2 2 .1" rgba=".4 .4 .4 1"/>
      <body pos="0 0 1"><freejoint/>
        <geom type="sphere" size=".1" rgba="1 .2 .1 1"/>
      </body>
    </worldbody></mujoco>
    ''')
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert data.time > 0 and data.qpos[2] < 1
    with mujoco.Renderer(model, height=240, width=320) as renderer:
        renderer.update_scene(data)
        frame = renderer.render()
        vendor = GL.glGetString(GL.GL_VENDOR).decode()
        device = GL.glGetString(GL.GL_RENDERER).decode()
        print(f'OpenGL vendor: {vendor}; renderer: {device}')
        assert 'nvidia' in vendor.lower(), f'Expected NVIDIA rendering, got {vendor}'
        assert frame.shape == (240, 320, 3) and np.ptp(frame) > 0
        iio.imwrite(output / 'mujoco.png', frame)
    print(f'MuJoCo {mujoco.__version__}: physics and NVIDIA EGL rendering passed')


def check_kitchen(output):
    import robocasa  # Registers tasks.
    from robocasa.utils.env_utils import create_env

    env = create_env(
        env_name='PickPlaceCounterToCabinet',
        robots='PandaOmron',
        layout_and_style_ids=[[1, 1]],
        seed=0,
        camera_names=['robot0_agentview_left'],
        camera_widths=320,
        camera_heights=240,
    )
    try:
        env.reset()
        low, high = env.action_spec
        action = np.clip(np.zeros_like(low), low, high)
        for _ in range(10):
            obs, _, _, _ = env.step(action)
        frame = obs['robot0_agentview_left_image']
        assert frame.shape == (240, 320, 3) and np.ptp(frame) > 0
        iio.imwrite(output / 'robocasa.png', frame[::-1])
        print(f'RoboCasa {robocasa.__version__}: kitchen reset, steps and camera passed')
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kitchen', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('/tmp/simulator-check'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    check_mujoco(args.output)
    if args.kitchen:
        check_kitchen(args.output)
    print(f'Images saved to {args.output}')
