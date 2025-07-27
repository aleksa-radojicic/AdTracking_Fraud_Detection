from typing import Annotated

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


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
