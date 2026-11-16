import os
import tempfile

import numpy as np
from ase.io import read

from deal import DataConfig, DEALConfig, SGPConfig, DEAL
from deal.model import DealActiveLearningModel


def main():
    frames = read("../data/traj.xyz", index=":5")

    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "deal")
        deal = DEAL(
            DataConfig(images=frames),
            DEALConfig(
                threshold=0.3,
                mask=False,
                output_prefix=prefix,
                save_gp=True,
                save_full_trajectory="final",
            ),
            SGPConfig(cutoff=4.5),
        )
        deal.run()

        # The saved GP must reload and reproduce the final-GP uncertainties.
        model = DealActiveLearningModel.from_file(f"{prefix}_sgp.json")
        assert model.training_size > 0
        for frame in read(f"{prefix}_trajectory_uncertainty.xyz", index=":"):
            uncertainty = model.predict_uncertainty(model.to_model_atoms(frame))
            np.testing.assert_allclose(
                uncertainty, frame.arrays["atomic_uncertainty"], atol=1e-6
            )

        # A full GP cannot be trained without force labels: fail clearly
        # instead of crashing in the C++ extension.
        unlabeled = [atoms.copy() for atoms in frames]
        deal = DEAL(
            DataConfig(images=unlabeled),
            DEALConfig(
                threshold=0.3,
                mask=False,
                output_prefix=os.path.join(tmp, "unlabeled"),
                save_gp=True,
            ),
            SGPConfig(cutoff=4.5),
        )
        try:
            deal.run()
        except ValueError as err:
            assert "no force labels" in str(err)
        else:
            raise AssertionError("save_gp with unlabeled frames did not raise")


if __name__ == "__main__":
    main()
