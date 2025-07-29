from typing import Annotated, Union

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame

from src.datatypes import BaseSchemaN, ExtendedSchema


class ClickTimestampInputS(pa.DataFrameModel):
    click_time: pl.Datetime('ms')


class ClickTimestampOutputS(pa.DataFrameModel):
    click_timestamp: pl.UInt32


@pa.check_types()
def make_click_timestamp_column(df: DataFrame[ClickTimestampInputS]) -> DataFrame[ClickTimestampOutputS]:
    I = ClickTimestampInputS
    O = ClickTimestampOutputS
    first_train_datetime = df.select(pl.col(I.click_time).min())

    click_timestamp = df.select(
        (pl.col(I.click_time)
        .dt.timestamp('ms')
        .sub(first_train_datetime.select(pl.col(I.click_time).dt.timestamp('ms')).row(0)[0])
        .floordiv(1000) # from ms to s
        )
        .cast(pl.UInt32)
        .alias(O.click_timestamp)
    )
    return click_timestamp


class PreviousSessionsInputS(pa.DataFrameModel):
    ip: pl.UInt32
    click_timestamp: pl.UInt32


class PreviousSessionsOutputS(pa.DataFrameModel):
    previous_sessions: pl.UInt32


@pa.check_types()
def make_previous_sessions_column(
    df: DataFrame[PreviousSessionsInputS],
    duration_between_sessions: Annotated[int, 's'] = 15 * 60,
) -> DataFrame[PreviousSessionsOutputS]:
    I = PreviousSessionsInputS
    O = PreviousSessionsOutputS
    previous_sessions = (df
    .select(
        pl.col(I.click_timestamp)
        .diff()
        .fill_null(0) # because of the 1st row (does not have previous row)
        .ge(duration_between_sessions)
        .cum_sum()
        .over(I.ip)
        .alias(O.previous_sessions)
    )
    )
    return previous_sessions


class TotalSessionsInputS(pa.DataFrameModel):
    ip: pl.UInt32
    previous_sessions: pl.UInt32


class TotalSessionsOutputS(pa.DataFrameModel):
    total_sessions: pl.UInt32


@pa.check_types()
def make_total_sessions_column(df: DataFrame[TotalSessionsInputS]) -> DataFrame[TotalSessionsOutputS]:
    I = TotalSessionsInputS
    O = TotalSessionsOutputS
    total_sessions = (
    df
    .select(pl.max(I.previous_sessions).over(I.ip).add(1).alias(O.total_sessions))
    )
    return total_sessions


class CurrentSessionDurationTillNowInputS(pa.DataFrameModel):
    ip: pl.UInt32
    previous_sessions: pl.UInt32
    click_timestamp: pl.UInt32


class CurrentSessionDurationTillNowOutputS(pa.DataFrameModel):
    current_session_duration_till_now: pl.UInt32


@pa.check_types()
def make_current_session_duration_till_now_column(df: DataFrame[CurrentSessionDurationTillNowInputS]) -> DataFrame[CurrentSessionDurationTillNowOutputS]:
    I = CurrentSessionDurationTillNowInputS
    O = CurrentSessionDurationTillNowOutputS
    current_session_duration_till_now = (df
    .sort(I.ip, I.previous_sessions, maintain_order=True)
    .select(
        pl.col(I.click_timestamp)
        .diff()
        .cum_sum()
        .over(I.ip, I.previous_sessions)
        .cast(pl.UInt32)
        .fill_null(0)
        .alias(O.current_session_duration_till_now)
        )
    )
    return current_session_duration_till_now


class CurrentSessionDurationInputS(pa.DataFrameModel):
    ip: pl.UInt32
    current_session_duration_till_now: pl.UInt32
    previous_sessions: pl.UInt32


class CurrentSessionDurationOutputS(pa.DataFrameModel):
    current_session_duration: pl.UInt32


@pa.check_types()
def make_current_session_duration_column(df: DataFrame[CurrentSessionDurationInputS]) -> DataFrame[CurrentSessionDurationOutputS]:
    I = CurrentSessionDurationInputS
    O = CurrentSessionDurationOutputS
    current_session_duration = (df
    .select(
        pl.col(I.current_session_duration_till_now).max().over(I.ip, I.previous_sessions).alias(O.current_session_duration)
    )
    )
    return current_session_duration


class AvgPreviousSessionsDurationInputS(pa.DataFrameModel):
    ip: pl.UInt32
    previous_sessions: pl.UInt32
    current_session_duration: pl.UInt32


class AvgPreviousSessionsDurationOutputS(pa.DataFrameModel):
    avg_previous_sessions_duration: pl.Float64


@pa.check_types()
def make_avg_previous_sessions_duration_column(df: DataFrame[AvgPreviousSessionsDurationInputS]) -> DataFrame[AvgPreviousSessionsDurationOutputS]:
    I = AvgPreviousSessionsDurationInputS
    O = AvgPreviousSessionsDurationOutputS
    avg_previous_sessions_duration_grouped = (
      df
      .group_by([I.ip, I.previous_sessions], maintain_order=True)
      .agg(pl.col(I.current_session_duration).max())
      .with_columns([
        pl.col(I.current_session_duration)
        .cum_sum()
        .over(I.ip)
        .alias("cum_prev_duration_w_current"),
      ])
      .with_columns(
        pl.col("cum_prev_duration_w_current")
        .sub(pl.col(I.current_session_duration))
        .alias("cum_prev_duration")
      )
      .with_columns(
        pl.col("cum_prev_duration")
        .truediv(pl.col(I.previous_sessions))
        .fill_nan(-1)
        .alias(O.avg_previous_sessions_duration)
      )
      .drop("cum_prev_duration_w_current", "cum_prev_duration")
    )

    avg_previous_sessions_duration = (df
        .join(
            avg_previous_sessions_duration_grouped,
            on=[I.ip, I.previous_sessions],
            how="left")
        .select(O.avg_previous_sessions_duration)
    )
    return avg_previous_sessions_duration


@pa.check_types(lazy=True)
def make_derived_columns(df: DataFrame[BaseSchemaN]) -> Union[DataFrame[BaseSchemaN], DataFrame[ExtendedSchema]]:
    click_timestamp = make_click_timestamp_column(df)
    df_extended = df.hstack(click_timestamp)
    previous_sessions = make_previous_sessions_column(df_extended)
    df_extended.hstack(previous_sessions, in_place=True)
    total_sessions = make_total_sessions_column(df_extended)
    df_extended.hstack(total_sessions, in_place=True)
    current_session_duration_till_now = make_current_session_duration_till_now_column(df_extended)
    df_extended.hstack(current_session_duration_till_now, in_place=True)
    current_session_duration = make_current_session_duration_column(df_extended)
    df_extended.hstack(current_session_duration, in_place=True)
    avg_previous_sessions_duration = make_avg_previous_sessions_duration_column(df_extended)
    df_extended.hstack(avg_previous_sessions_duration, in_place=True)
    return df_extended
