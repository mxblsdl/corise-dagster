import csv
from datetime import datetime
from typing import List
from heapq import nlargest

from dagster import In, Out, String, job, op, usable_as_dagster_type
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


def csv_helper(file_name: str):
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            yield Stock.from_list(row)


@op(config_schema={"s3_key": String}, out={"out": Out(dagster_type=List[Stock])})
def get_s3_data_op(context):
    s3_loc = context.op_config["s3_key"]
    stocks = csv_helper(s3_loc)
    return list(stocks)


# Helper for nlargest sort
def sortkey(stock):
    return stock.high


@op(ins={"data": In(dagster_type=List[Stock])}, out={"out": Out(dagster_type=Aggregation)})
def process_data_op(context, data):
    stock = nlargest(1, data, key=sortkey)
    agg = Aggregation(date=stock[0].date, high=stock[0].high)
    return agg


@op
def put_redis_data_op(context, aggregation):
    pass


@op
def put_s3_data_op(context, aggregation):
    pass


@job
def machine_learning_job():
    stocks = get_s3_data_op()
    data = process_data_op(data=stocks)
    put_redis_data_op(aggregation=data)
    put_s3_data_op(aggregation=data)
