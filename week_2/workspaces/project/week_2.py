from datetime import datetime
from typing import List
from heapq import nlargest
from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    ResourceDefinition,
    String,
    graph,
    op,
)
from workspaces.config import REDIS, S3, S3_FILE
from workspaces.resources import mock_s3_resource, redis_resource, s3_resource
from workspaces.types import Aggregation, Stock


@op(config_schema={"s3_key": String}, required_resource_keys={"s3"}, out={"s3_out": Out(dagster_type=List[Stock])})
def get_s3_data(context: OpExecutionContext):
    out = context.resources.s3.get_data(key_name=context.op_config["s3_key"])
    return [Stock.from_list(o) for o in out]


# Helper for nlargest sort
def sortkey(stock: Stock) -> int:
    return stock.high


@op(ins={"process_data": In(dagster_type=List[Stock])}, out={"out": Out(dagster_type=Aggregation)})
def process_data(context, process_data):
    stock = nlargest(1, process_data, key=sortkey)
    return Aggregation(date=stock[0].date, high=stock[0].high)


@op(required_resource_keys={"redis"}, ins={"agg": In(dagster_type=Aggregation)}, out=Out(Nothing))
def put_redis_data(context, agg):
    context.resources.redis.put_data(name=agg.date, value=str(agg.high))


@op(
    required_resource_keys={"s3"},
    ins={"agg": In(dagster_type=Aggregation)},
    out=Out(Nothing),
    description="Put aggregation data in S3 bucket",
)
def put_s3_data(context, agg):
    context.resources.s3.put_data(key_name=agg.date, data=agg)


@graph
def machine_learning_graph():
    stocks = get_s3_data()
    data = process_data(process_data=stocks)
    put_redis_data(agg=data)
    put_s3_data(agg=data)


local = {
    "ops": {"get_s3_data": {"config": {"s3_key": S3_FILE}}},
}

docker = {
    "resources": {
        "s3": {"config": S3},
        "redis": {"config": REDIS},
    },
    "ops": {"get_s3_data": {"config": {"s3_key": S3_FILE}}},
}

machine_learning_job_local = machine_learning_graph.to_job(
    name="machine_learning_job_local",
    config=local,
    resource_defs={"s3": ResourceDefinition.mock_resource(), "redis": ResourceDefinition.mock_resource()},
)

machine_learning_job_docker = machine_learning_graph.to_job(
    name="machine_learning_job_docker", config=docker, resource_defs={"s3": s3_resource, "redis": redis_resource}
)
