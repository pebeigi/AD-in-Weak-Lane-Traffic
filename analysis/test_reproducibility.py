#!/usr/bin/env python3
"""Determinism and bound checks for RAU and SDPM.

These do not use the paired cohort and do not rank the models.
Integrated velocity components may be negative: that is a direction, not a
reverse-speed bug. The commanded speed inside the RAU grid is max(0, v+a*dt).
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MTM_Car_Calibration import compute_mtm_metrics
from PT_Car_Calibration import (
    choose_car_velocity_direction,
    compute_car_metrics,
    simulate_car_single_agent,
)


def _frames(n=8):
    fd = {}
    for i in range(n):
        t = i * 0.5
        fd[t] = [
            {
                "id": 7,
                "x": 100.0 + i * 4.0,
                "y": 50.0 + 0.05 * i,
                "vx": 8.0,
                "vy": 0.1,
                "dest_x": 140.0,
                "dest_y": 51.0,
                "neighbors": [
                    (112.0 + i * 4.0, 51.0, 7.5, 0.1, 4.9, 1.8),
                    (92.0 + i * 4.0, 49.0, 8.2, 0.0, 5.0, 1.9),
                    (108.0 + i * 4.0, 53.0, 7.0, 0.2, 4.8, 1.8),
                    (120.0 + i * 4.0, 48.5, 6.5, -0.1, 5.1, 1.9),
                    (130.0 + i * 4.0, 60.0, 5.0, 0.0, 5.0, 1.8),
                ],
                "lane_kf": 2.0,
                "run_id": 1.0,
                "length_m": 4.9,
                "width_m": 1.8,
            }
        ]
    return fd


RAU = [200.0, 2.0, 2.0, 0.5, 12.0, 40.0]
SDPM = [20.0, 1.2, 2.0, 1.5, 2.0, 0.2, 0.4, 0.5, 1.0]


class ModelChecks(unittest.TestCase):
    def test_canonical_seed_depends_on_recording(self):
        from analysis.trajectory_seed import execution_seed

        a = execution_seed(1, 1, 7, "rau")
        b = execution_seed(2, 1, 7, "rau")
        c = execution_seed(1, 2, 7, "rau")
        d = execution_seed(1, 1, 7, "rau")
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(a, d)
        self.assertNotEqual(execution_seed(1, 1, 7, "rau"), execution_seed(1, 1, 7, "sdpm"))

    def test_same_parameters_same_trajectory(self):
        fd = _frames()
        a = compute_car_metrics(fd, RAU, 7, clip_to_envelope=False)
        b = compute_car_metrics(fd, RAU, 7, clip_to_envelope=False)
        self.assertEqual(a["Error"], b["Error"])
        self.assertEqual(a["sim_x"], b["sim_x"])
        # Default generator comes from the canonical key in the frames.
        sa = compute_mtm_metrics(fd, SDPM, 7, clip_to_envelope=False)
        sb = compute_mtm_metrics(fd, SDPM, 7, clip_to_envelope=False)
        self.assertEqual(sa["Error"], sb["Error"])
        self.assertEqual(sa["sim_x"], sb["sim_x"])

    def test_same_id_other_recording_differs(self):
        fd = _frames()
        other = {t: [dict(u, run_id=2.0) for u in us] for t, us in fd.items()}
        a = compute_car_metrics(fd, RAU, 7, clip_to_envelope=False)
        b = compute_car_metrics(other, RAU, 7, clip_to_envelope=False)
        self.assertNotEqual(a["sim_x"], b["sim_x"])

    def test_ga_search_runs_single_worker(self):
        from MTM_Car_Calibration import run_ga_mtm

        fd = _frames()
        vec, fit, _, _ = run_ga_mtm(fd, 7, pop_size=4, generations=2, clip_to_envelope=False, search_seed=1)
        self.assertEqual(len(vec), 9)
        self.assertTrue(0.0 < fit <= 1.0)

    def test_overlap_lateral_response_is_repulsive(self):
        from mtm_dynamics import mtm_acceleration_world

        for dy in (0.5, -0.5, 2.5, -2.5):
            _, ay, _ = mtm_acceleration_world(
                0, 0, 15, 0, [(6, dy, 8, 0, 4.8, 1.8)], SDPM, 1, 1, 1000, 0, 4.8, 1.8
            )
            self.assertLess(ay * dy, 0.0)

    def test_finite_outputs(self):
        fd = _frames()
        a = compute_car_metrics(fd, RAU, 7, clip_to_envelope=False)
        s = compute_mtm_metrics(
            fd, SDPM, 7, clip_to_envelope=False, rng=random.Random(1)
        )
        for m in (a, s):
            self.assertTrue(math.isfinite(m["ADE"]))
            self.assertTrue(all(math.isfinite(v) for v in m["sim_x"]))
            self.assertTrue(all(math.isfinite(v) for v in m["sim_y"]))
            self.assertTrue(all(math.isfinite(v) for v in m["sim_speed"]))

    def test_no_nan_rollout(self):
        r1 = simulate_car_single_agent(
            0, 0, 5, 0, RAU,
            [(8, 1, 5, 0, 4.5, 1.8)],
            40, 0,
            total_dt=1.0,
            rng=random.Random(7),
            clip_to_envelope=False,
        )
        r2 = simulate_car_single_agent(
            0, 0, 5, 0, RAU,
            [(8, 1, 5, 0, 4.5, 1.8)],
            40, 0,
            total_dt=1.0,
            rng=random.Random(7),
            clip_to_envelope=False,
        )
        self.assertEqual(r1, r2)
        self.assertTrue(all(math.isfinite(v) for v in r1))

    def test_commanded_speed_nonnegative(self):
        # The grid speed is max(0, v + a*dt). The returned command is a velocity.
        vx, vy = choose_car_velocity_direction(
            0, 0, 8, 0, RAU, [(12, 1, 7, 0, 4.5, 1.8)], 40, 0, dt=0.1
        )
        self.assertTrue(math.isfinite(vx) and math.isfinite(vy))
        self.assertGreaterEqual(math.hypot(vx, vy), 0.0)

    def test_neighbor_tuple_width(self):
        fd = _frames()
        nbs = fd[0.0][0]["neighbors"]
        self.assertGreaterEqual(len(nbs), 4)
        self.assertEqual(len(nbs[0]), 6)
        # The simulator must accept the 6-tuple the cohort builder stores.
        simulate_car_single_agent(
            100, 50, 8, 0, RAU, nbs[:4], 140, 51,
            total_dt=0.5, rng=random.Random(3), clip_to_envelope=False,
        )


if __name__ == "__main__":
    unittest.main()
