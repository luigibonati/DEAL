from deal.runtime import collect_runtime_info, format_runtime_info


def main():
    info = collect_runtime_info()
    output = format_runtime_info(info)

    assert info["logical_cpus"] >= 1
    assert info["available_cpus"] >= 1
    assert "openmp_enabled" in info
    assert "[INFO] Runtime environment:" in output
    assert "- CPU:" in output
    assert "- OpenMP:" in output
    assert "OMP_NUM_THREADS=" in output
    assert "OPENBLAS_NUM_THREADS=" in output


if __name__ == "__main__":
    main()
