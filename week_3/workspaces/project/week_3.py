from datetime import datetime
from typing import List
from heapq import nlargest

from dagster import (
    In,
    Nothing,
    String,
    OpExecutionContext,
    Out,
    ResourceDefinition,
    RetryPolicy,
    RunRequest,
    ScheduleDefinition,
    SensorEvaluationContext,
    SkipReason,
    graph,
    op,
    schedule,
    sensor,
    static_partitioned_config,
)
from workspaces.config import REDIS, S3
from workspaces.project.sensors import get_s3_keys
from workspaces.resources import mock_s3_resource, redis_resource, s3_resource
from workspaces.types import Aggregation, Stock


@op(
    config_schema={"s3_key": String},
    required_resource_keys={"s3"},
    out={"s3_out": Out(dagster_type=List[Stock])},
    description="Pull data from S3 and convert to list of Stocks",
)
def get_s3_data(context: OpExecutionContext):
    out = context.resources.s3.get_data(key_name=context.op_config["s3_key"])
    return [Stock.from_list(o) for o in out]


# Helper for nlargest sort
def sortkey(stock: Stock) -> int:
    return stock.high


@op(
    ins={"process_data": In(dagster_type=List[Stock])},
    out={"out": Out(dagster_type=Aggregation)},
    description="Take stock data and find n largest",
)
def process_data(context, process_data):
    stock = nlargest(1, process_data, key=sortkey)
    return Aggregation(date=stock[0].date, high=stock[0].high)


@op(
    required_resource_keys={"redis"},
    ins={"agg": In(dagster_type=Aggregation)},
    out=Out(Nothing),
    description="Load data to redis",
)
def put_redis_data(context, agg):
    context.resources.redis.put_data(name=str(agg.date), value=str(agg.high))


@op(
    required_resource_keys={"s3"},
    ins={"agg": In(dagster_type=Aggregation)},
    out=Out(Nothing),
    description="Put aggregation data in S3 bucket",
)
def put_s3_data(context, agg):
    context.resources.s3.put_data(key_name=str(agg.date), data=agg)


@graph
def machine_learning_graph():
    data = process_data(get_s3_data())

    put_redis_data(agg=data)
    put_s3_data(agg=data)


local = {
    "ops": {"get_s3_data": {"config": {"s3_key": "prefix/stock_9.csv"}}},
}


docker = {
    "resources": {
        "s3": {"config": S3},
        "redis": {"config": REDIS},
    },
    "ops": {"get_s3_data": {"config": {"s3_key": "prefix/stock_9.csv"}}},
}


@static_partitioned_config(partition_keys=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
def docker_config(partition_key: str):
    return {
        "resources": {"s3": {"config": S3}, "redis": {"config": REDIS}},
        "ops": {"get_s3_data": {"config": {"s3_key": f"prefix/stock_{partition_key}.csv"}}},
    }


machine_learning_job_local = machine_learning_graph.to_job(
    name="machine_learning_job_local",
    config=local,
    resource_defs={"s3": mock_s3_resource, "redis": ResourceDefinition.mock_resource()},
)

machine_learning_job_docker = machine_learning_graph.to_job(
    name="machine_learning_job_docker",
    resource_defs={"s3": s3_resource, "redis": redis_resource},
    op_retry_policy=RetryPolicy(max_retries=10, delay=1),
    config=docker,
)


machine_learning_schedule_local = ScheduleDefinition(job=machine_learning_job_local, cron_schedule="*/15 * * * *")


@schedule(cron_schedule="0 * * * *", pipeline_name="machine_learning_job_docker", job=machine_learning_job_docker)
def machine_learning_schedule_docker():
    pass


@sensor(job=machine_learning_job_docker, minimum_interval_seconds=30)
def machine_learning_sensor_docker(context):
    new_files = get_s3_keys(bucket="dagster", prefix="prefix", endpoint_url="http://localstack:4566")
    if not new_files:
        yield SkipReason("No new s3 files found in bucket.")
        return
    for new_file in new_files:
        yield RunRequest(
            run_key=new_file,
            run_config={
                "resources": {
                    "s3": {"config": S3},
                    "redis": {"config": REDIS},
                },
                "ops": {"get_s3_data": {"config": {"s3_key": new_file}}},
            },
        )
