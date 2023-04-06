import csv
from datetime import datetime
from heapq import nlargest
from typing import Iterator, List

from dagster import (
    Any,
    DynamicOut,
    DynamicOutput,
    In,
    Nothing,
    OpExecutionContext,
    Out,
    Output,
    String,
    job,
    op,
    usable_as_dagster_type,
)
from pydantic import BaseModel


@usable_as_dagster_type(description="Stock data")
class Stock(BaseModel):
    date: datetime
    close: float
    volume: int
    open: float
    high: float
    low: float

    @classmethod
    def from_list(cls, input_list: List[str]):
        """Do not worry about this class method for now"""
        return cls(
            date=datetime.strptime(input_list[0], "%Y/%m/%d"),
            close=float(input_list[1]),
            volume=int(float(input_list[2])),
            open=float(input_list[3]),
            high=float(input_list[4]),
            low=float(input_list[5]),
        )


@usable_as_dagster_type(description="Aggregation of stock data")
class Aggregation(BaseModel):
    date: datetime
    high: float


def csv_helper(file_name: str) -> Iterator[Stock]:
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            yield Stock.from_list(row)


@op(
    config_schema={"s3_key": String},
    out={
        "stocks": Out(dagster_type=List[Stock], is_required=False),
        "empty_stocks": Out(is_required=False),
    },
)
def get_s3_data_op(context):
    s3_loc = context.op_config["s3_key"]
    stocks = list(csv_helper(s3_loc))
    if len(stocks) == 0:
        yield Output(None, "empty_stocks")
    else:
        yield Output(stocks, "stocks")


# Helper for nlargest sort
def sortkey(stock):
    return stock.high


@op(ins={"data": In(dagster_type=List[Stock])}, out=DynamicOut())
def process_data_op(context, data):
    stocks = nlargest(context.op_config["nlargest"], data, key=sortkey)

    aggregates = [Aggregation(date=s.date, high=s.high) for s in stocks]
    for k, v in enumerate(aggregates):
        yield DynamicOutput(v, mapping_key=str(k))


@op
def put_redis_data_op(context, aggregation):
    pass


@op
def put_s3_data_op(context, aggregation):
    pass


@op(
    ins={"empty_stocks": In(dagster_type=Any)},
    out=Out(Nothing),
    description="Notifiy if stock list is empty",
)
def empty_stock_notify_op(context: OpExecutionContext, empty_stocks: Any):
    context.log.info("No stocks returned")


@job
def machine_learning_dynamic_job():
    stocks, empty_stocks = get_s3_data_op()
    data = process_data_op(data=stocks)

    empty_stock_notify_op(empty_stocks)

    datas = data.map(put_redis_data_op)
    datas.collect()

    datas = data.map(put_s3_data_op)
    datas.collect()
