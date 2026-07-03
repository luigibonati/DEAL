import yaml

from deal import DEALConfig
from deal.cli import _parse_mask_arg, _resolve_default_mask


def main():
    default = DEALConfig(threshold=0.5)
    assert default.mask is None
    assert default.update_threshold == 0.4

    # Lowercase ``none`` is parsed by PyYAML as a string. Keep accepting the
    # spelling used by older DEAL templates and apply the computed default.
    parsed = yaml.safe_load("update_threshold: none")
    config = DEALConfig(threshold=0.5, **parsed)
    assert config.update_threshold == 0.4

    explicit_mask = DEALConfig(mask=True)
    assert explicit_mask.mask is True
    assert _parse_mask_arg("null") is None

    config = {
        "deal": {"mask": None},
        "preprocessing": {"mask_key": "custom_mask"},
    }
    _resolve_default_mask(config)
    assert config["deal"]["mask"] == "custom_mask"

    explicit_false = {
        "deal": {"mask": False},
        "preprocessing": {"mask_key": "custom_mask"},
    }
    _resolve_default_mask(explicit_false)
    assert explicit_false["deal"]["mask"] is False


if __name__ == "__main__":
    main()
