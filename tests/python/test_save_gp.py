import os
import tempfile

import numpy as np
from ase.io import read

from deal import DataConfig, DEALConfig, SGPConfig, DEAL
from deal.model import DealActiveLearningModel


def main():
    frames = read("../data/traj.xyz", index=":5")

    unlabeled = [atoms.copy() for atoms in frames]

    with tempfile.TemporaryDirectory() as tmp:
        # Labeled or not, the saved GP is uncertainty-only (fast path) and
        # must reload and reproduce the final-GP uncertainties.
        for name, images in [("labeled", frames), ("unlabeled", unlabeled)]:
            prefix = os.path.join(tmp, name)
            deal = DEAL(
                DataConfig(images=images),
                DEALConfig(
                    threshold=0.3,
                    mask=False,
                    output_prefix=prefix,
                    save_gp=True,
                    save_full_trajectory="final",
                ),
                SGPConfig(cutoff=4.5),
            )
            assert deal._use_local_uncertainty_fast_path()
            deal.run()
            model = DealActiveLearningModel.from_file(f"{prefix}_sgp.json")
            assert len(model.gp.local_environments) > 0
            for frame in read(f"{prefix}_trajectory_uncertainty.xyz", index=":"):
                uncertainty = model.predict_uncertainty(model.to_model_atoms(frame))
                np.testing.assert_allclose(
                    uncertainty, frame.arrays["atomic_uncertainty"], atol=1e-6
                )

        # train_hyps builds the full GP (labels, posterior mean): it must
        # also reload and reproduce the final-GP uncertainties.
        prefix = os.path.join(tmp, "labeled_hyps")
        deal = DEAL(
            DataConfig(images=frames),
            DEALConfig(
                threshold=0.3,
                mask=False,
                output_prefix=prefix,
                train_hyps=True,
                save_gp=True,
                save_full_trajectory="final",
            ),
            SGPConfig(cutoff=4.5),
        )
        assert not deal._use_local_uncertainty_fast_path()
        deal.run()
        model = DealActiveLearningModel.from_file(f"{prefix}_sgp.json")
        assert model.training_size > 0
        for frame in read(f"{prefix}_trajectory_uncertainty.xyz", index=":"):
            uncertainty = model.predict_uncertainty(model.to_model_atoms(frame))
            np.testing.assert_allclose(
                uncertainty, frame.arrays["atomic_uncertainty"], atol=1e-6
            )

        # Hyperparameter training needs a full GP, hence force labels: fail
        # clearly instead of crashing in the C++ extension.
        deal = DEAL(
            DataConfig(images=unlabeled),
            DEALConfig(
                threshold=0.3,
                mask=False,
                output_prefix=os.path.join(tmp, "unlabeled_hyps"),
                train_hyps=True,
            ),
            SGPConfig(cutoff=4.5),
        )
        try:
            deal.run()
        except ValueError as err:
            assert "no force labels" in str(err)
        else:
            raise AssertionError("train_hyps with unlabeled frames did not raise")

if __name__ == "__main__":
    main()
