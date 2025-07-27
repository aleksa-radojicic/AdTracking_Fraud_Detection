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
