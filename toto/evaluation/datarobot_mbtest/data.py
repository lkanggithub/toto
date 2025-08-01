import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import List
from typing import Iterable, Iterator

from datasets import Dataset
from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import TestData, TrainingDataset, split
from gluonts.itertools import Map
from gluonts.transform import Transformation
from gluonts.dataset.field_names import FieldName
import pyarrow.compute as pc
from toolz import compose
import pandas as pd
import numpy as np

from dr_model_benchmark.common.entities import DataRobotMBTestDatasetConfig


MAX_WINDOW = 20


def itemize_start(data_entry: DataEntry) -> DataEntry:
    data_entry["start"] = data_entry["start"].item()
    return data_entry


class MultivariateToUnivariate(Transformation):
    def __init__(self, field):
        self.field = field

    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool = False
    ) -> Iterator:
        for data_entry in data_it:
            item_id = data_entry["item_id"]
            val_ls = list(data_entry[self.field])
            for id, val in enumerate(val_ls):
                univariate_entry = data_entry.copy()
                univariate_entry[self.field] = val
                univariate_entry["item_id"] = item_id + "_dim" + str(id)
                yield univariate_entry


def get_dataset_name(dataset_path: Path) -> str:  # FIXME
    dataset_file_name = dataset_path.name
    return dataset_file_name.split("_train.csv")[0]


def sort_dataframe_by_datetime_column(
    data_frame: pd.DataFrame,
    datetime_column_name: str,
) -> pd.DataFrame:
    datatime_column = pd.to_datetime(data_frame[datetime_column_name])
    sort_column_name = f"{datetime_column_name}_for_sort"
    data_frame[sort_column_name] = datatime_column
    sorted_data_frame = data_frame.sort_values(by=[sort_column_name])
    sorted_data_frame.drop([sort_column_name], axis=1, inplace=True)
    return sorted_data_frame


def get_hf_dataset_target_dimension(hf_dataset: Dataset) -> int:
    return (
        target.shape[0]
        if len((target := hf_dataset[0][FieldName.TARGET]).shape) > 1
        else 1
    )


def create_one_univariate_dataset(
    dataframe: pd.DataFrame,
    datetime_column_name: str,
    variate_column_name: str,
) -> Dataset:

    data_dict = {
        FieldName.START: np.array([pd.to_datetime(dataframe[datetime_column_name]).iloc[0]]),
        # FieldName.TARGET: np.array([dataframe[variate_column_name].values]),
        FieldName.TARGET: dataframe[variate_column_name].values.reshape(-1, 1).transpose(1, 0),
    }
    return Dataset.from_dict(data_dict).with_format("numpy")


def create_one_multivariate_dataset(
    dataframe: pd.DataFrame,
    datetime_column_name: str,
    target_column_name: str,
) -> Dataset:
    covariate_column_names = [
        column
        for column in dataframe.columns
        if column not in {datetime_column_name, target_column_name}
    ]
    data_dict = {
        FieldName.START: np.array([pd.to_datetime(dataframe[datetime_column_name]).iloc[0]]),
        FieldName.TARGET: dataframe[target_column_name].values.reshape(-1, 1).transpose(1, 0),
    }
    numeric_cols = [
        col
        for col in covariate_column_names
        if dataframe[col].dtype.kind in "iufcb"
    ]
    non_numeric_cols = [
        col
        for col in covariate_column_names
        if dataframe[col].dtype.kind not in "iufcb"
    ]
    if numeric_cols:
        data_dict.update(
            {FieldName.PAST_FEAT_DYNAMIC_REAL: dataframe[numeric_cols].values.transpose(1, 0)}
        )
    if non_numeric_cols:
        data_dict.update(
            {FieldName.PAST_FEAT_DYNAMIC_CAT: dataframe[non_numeric_cols].values.transpose(1, 0)}
        )

    return _FileDataset(
        data_dict, freq=self.time_series_frequency, one_dim_target=self.target_dim == 1,
    ).with_format("numpy")
    # return Dataset.from_dict(data_dict).with_format("numpy")


def create_one_multiseries_dataset(
    dataframe: pd.DataFrame,
    datetime_column_name: str,
) -> Dataset:
    # TODO
    pass


def create_one_gluonts_dataset_from_hf_dataset(
    hf_dataset: Dataset,
    time_series_frequency: str,
    to_univariate_gluonts_dataset: bool = False,
) -> object:  # FIXME
    process = ProcessDataEntry(
        time_series_frequency,
        one_dim_target=True,
    )
    gluonts_dataset = Map(compose(process, itemize_start), hf_dataset)
    if to_univariate_gluonts_dataset:
        gluonts_dataset = MultivariateToUnivariate(FieldName.TARGET).apply(
            gluonts_dataset
        )
    return gluonts_dataset


@dataclass
class TestDataset:
    name: str
    hf_dataset: Dataset
    gluonts_dataset: object
    prediction_length: int
    test_data_length: int
    train_end_date: str
    time_series_frequency: str

    @cached_property
    def target_dim(self) -> int:
        return get_hf_dataset_target_dimension(self.hf_dataset)

    @cached_property
    def past_feat_dynamic_real_dim(self) -> int:

        if FieldName.PAST_FEAT_DYNAMIC_REAL not in self.hf_dataset[0]:
            return 0
        elif (
            len(
                (
                    past_feat_dynamic_real := self.hf_dataset[0][
                        FieldName.PAST_FEAT_DYNAMIC_REAL
                    ]
                ).shape
            )
            > 1
        ):
            return past_feat_dynamic_real.shape[0]
        else:
            return 1

    @cached_property
    def windows(self) -> int:
        w = math.floor(self.test_data_length / self.prediction_length)
        return min(max(1, w), MAX_WINDOW)

    @cached_property
    def _min_series_length(self) -> int:
        if self.hf_dataset[0][FieldName.TARGET].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(
                    pc.list_slice(self.hf_dataset.data.column(FieldName.TARGET), 0, 1)
                )
            )
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column(FieldName.TARGET))
        return min(lengths.to_numpy())

    @cached_property
    def sum_series_length(self) -> int:
        if self.hf_dataset[0][FieldName.TARGET].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(self.hf_dataset.data.column(FieldName.TARGET))
            )
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column(FieldName.TARGET))
        return sum(lengths.to_numpy())

    @cached_property
    def split_point(self) -> pd.Period:
        return pd.Period(self.train_end_date, freq=self.time_series_frequency)

    @property
    def training_dataset(self) -> TrainingDataset:
        training_dataset, _ = split(
            self.gluonts_dataset,
            date=self.split_point,
        )
        return training_dataset

    @property
    def test_data(self) -> TestData:
        _, test_template = split(
            self.gluonts_dataset,
            date=self.split_point,
        )
        test_data = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )
        return test_data

    @classmethod
    def create_from_datarobot_mbtest_yaml(
        cls,
        datarobot_mbtest_yaml_path: Path,
        time_series_frequency: str,
        create_univariate_dataset: bool = False,
    ) -> List["TestDataset"]:
        test_datasets: List["TestDataset"] = []
        mbtest_configs = DataRobotMBTestDatasetConfig.load_from_yaml(datarobot_mbtest_yaml_path)
        for mbtest_config in mbtest_configs:
            time_series_config = mbtest_config.time_series_config
            datetime_partition_column_name = mbtest_config.partitioning.partition_column
            train_dataframe = pd.read_csv(mbtest_config.train_dataset_path)
            test_dataframe = pd.read_csv(mbtest_config.pred_dataset_path)
            full_dataframe = pd.concat([train_dataframe, test_dataframe], axis=0, ignore_index=True)
            full_dataframe = sort_dataframe_by_datetime_column(
                full_dataframe, datetime_partition_column_name,
            )
            dataset_name = get_dataset_name(Path(mbtest_config.train_dataset_path))
            if create_univariate_dataset and dataset_name.find("univariate") != -1:  # FIXME
                hf_dataset = create_one_univariate_dataset(
                    dataframe=full_dataframe,
                    datetime_column_name=datetime_partition_column_name,
                    variate_column_name=mbtest_config.target,
                )
            else:
                hf_dataset = create_one_multivariate_dataset(
                    dataframe=full_dataframe,
                    datetime_column_name=datetime_partition_column_name,
                    target_column_name=mbtest_config.target,
                )
            test_datasets.append(
                cls(
                    name=dataset_name,
                    hf_dataset=hf_dataset,
                    gluonts_dataset=create_one_gluonts_dataset_from_hf_dataset(
                        hf_dataset, time_series_frequency,
                    ),
                    prediction_length=time_series_config.forecast_window_end - time_series_config.forecast_window_start + 1,
                    test_data_length=len(test_dataframe),
                    train_end_date=train_dataframe[datetime_partition_column_name].values[-1],
                    time_series_frequency=time_series_frequency,
                )
            )

        return test_datasets
