import numpy as np

from deal.sgp.atoms import SGPAtoms, Trajectory


frames = [
    SGPAtoms(numbers=[1], positions=[[float(index), 0.0, 0.0]])
    for index in range(3)
]
original_positions = [frame.positions.copy() for frame in frames]

np.random.seed(42)
trajectory = Trajectory(frames, iterate_strategy="shuffle")

assert len(trajectory) == len(frames)
assert trajectory.frames is not frames
assert all(
    np.array_equal(frame.positions, positions)
    for frame, positions in zip(frames, original_positions)
)

force_frame = SGPAtoms(numbers=[1, 1], positions=np.zeros((2, 3)))
forces = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
force_frame.forces = forces
force_trajectory = Trajectory([force_frame])

np.testing.assert_array_equal(force_trajectory.get_next_force(), forces)
np.testing.assert_array_equal(force_trajectory.get_next_force(1), forces[1])
