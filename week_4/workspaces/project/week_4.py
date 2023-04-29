from datetime import datetime
from typing import List

from dagster import (
    AssetSelection,
    Nothing,
    OpExecutionContext,
    ScheduleDefinition,
    String,
    asset,
    define_asset_job,
    load_assets_from_current_module,
    static_partitioned_config,
    StaticPartitionsDefinition,
)
from workspaces.types import Aggregation, Stock


@asset(
    config_schema={"s3_key": String},
    required_resource_keys={"s3"},
)
def get_s3_data(context: OpExecutionContext) -> List[Stock]:
    out = context.resources.s3.get_data(key_name=context.op_config["s3_key"])
    return [Stock.from_list(o) for o in out]


@asset()
def process_data(context, get_s3_data) -> Aggregation:
    stock = max(get_s3_data, key=lambda x: x.high)
    return Aggregation(date=stock.date, high=stock.high)


@asset(required_resource_keys={"redis"}, op_tags={"kind": "redis"})
def put_redis_data(context, process_data):
    context.resources.redis.put_data(name=str(process_data.date), value=str(process_data.high))


@asset(required_resource_keys={"s3"}, op_tags={"kind": "s3"})
def put_s3_data(context, process_data):
    context.resources.s3.put_data(key_name=str(process_data.date), data=process_data)


project_assets = load_assets_from_current_module(group_name="dagster")


machine_learning_asset_job = define_asset_job(
    name="machine_learning_asset_job",
    selection=AssetSelection.groups("dagster"),
    config={"ops": {"get_s3_data": {"config": {"s3_key": f"prefix/stock_3.csv"}}}},
)

machine_learning_schedule = ScheduleDefinition(job=machine_learning_asset_job, cron_schedule="*/15 * * * *")
