from pathlib import Path

from toto.evaluation.datarobot_mbtest.data import TestDataset


def test():
    from pathlib import Path
    from toto.evaluation.datarobot_mbtest.data import TestDataset

    datasets = TestDataset.create_from_datarobot_mbtest_yaml(
        datarobot_mbtest_yaml_path=Path(
            "/home/lyndon.kang/projects/foundation_model_compare/mbtest/"
            "custom_data_no_pii_ts_with_local_path.yaml"
        ),
        time_series_frequency="D",  # FIXME
        create_univariate_dataset=True,
    )
    import pdb
    pdb.set_trace()
    assert False
