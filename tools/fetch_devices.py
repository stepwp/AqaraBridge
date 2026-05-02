#!/usr/bin/env python3
"""采集 Aqara 设备数据，保存为 sample JSON 文件

对于 query.resource.info 返回空的设备（老型号/子设备），
自动 fallback 到用常见资源 ID 探测。

用法:
    python tools/fetch_devices.py                          # 增量采集（跳过已有 sample）
    python tools/fetch_devices.py --force                  # 强制重新采集所有设备
    python tools/fetch_devices.py --model lumi.switch.acn056  # 只采集指定 model
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import yaml

from aqara_client import AqaraClient, APIError

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
CONFIG_PATH = os.path.join(TOOLS_DIR, "config.yaml")
SAMPLES_DIR = os.path.join(TOOLS_DIR, "samples")
DEVICE_LIST_PATH = os.path.join(PROJECT_DIR, "docs", "aqara_device_list.md")

# 所有已知的 Aqara 资源 ID（从 aiot_mapping.py 和文档中汇总）
# 用于在 query.resource.info 返回空时做探测
ALL_KNOWN_RESOURCE_IDS = [
    # 传感器
    "0.1.85", "0.2.85", "0.3.85", "0.4.85", "0.5.85",
    # 功率/能耗
    "0.11.85", "0.12.85", "0.13.85", "0.14.85",
    # 旋转角度
    "0.22.85", "0.29.85",
    # 位置/亮度/色温
    "1.1.85", "1.7.85", "1.8.85", "1.9.85", "1.10.85",
    # 运动/存在
    "3.1.85", "3.2.85", "3.51.85",
    # 开关控制
    "4.1.85", "4.2.85", "4.3.85", "4.4.85",
    "4.21.85", "4.22.85", "4.66.85", "4.67.85",
    # 事件
    "13.1.85", "13.2.85", "13.3.85", "13.4.85",
    "13.12.85", "13.21.85", "13.22.85", "13.23.85", "13.24.85",
    "13.27.85",
    # 灯光
    "14.1.85", "14.2.85", "14.4.85", "14.7.85", "14.7.111", "14.7.1006",
    "14.8.85", "14.10.85", "14.32.85", "14.35.85", "14.51.85", "14.140.85",
    # 系统
    "8.0.2001", "8.0.2007", "8.0.2008", "8.0.2115", "8.0.2116",
]


def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"配置文件不存在: {CONFIG_PATH}")
        print("请先运行: python tools/auth.py")
        sys.exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not config.get("access_token"):
        print("access_token 为空，请先运行: python tools/auth.py")
        sys.exit(1)
    return config


def load_official_names():
    """从 aqara_device_list.md 加载 model -> 官方产品名"""
    names = {}
    if not os.path.exists(DEVICE_LIST_PATH):
        return names
    with open(DEVICE_LIST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if "|" not in line or "---" in line or "产品名称" in line or "设备品类" in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            parts = [p for p in parts if p]
            if len(parts) >= 2 and "." in parts[1]:
                names[parts[1]] = parts[0]
    return names


def load_existing_samples():
    """加载已有的 sample，返回 {model: has_resources}"""
    existing = {}
    if not os.path.isdir(SAMPLES_DIR):
        return existing
    for fname in os.listdir(SAMPLES_DIR):
        if not fname.endswith(".json"):
            continue
        filepath = os.path.join(SAMPLES_DIR, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = fname.replace(".json", "")
        existing[model] = bool(data.get("resources"))
    return existing


def save_sample(model, data):
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    filepath = os.path.join(SAMPLES_DIR, f"{model}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filepath


def fetch_resource_info(client, model):
    """获取 model 支持的所有资源定义"""
    try:
        result = client.query_resource_info(model)
        if not result:
            return {}
        resources = {}
        data_list = result if isinstance(result, list) else result.get("data", [])
        if isinstance(data_list, dict):
            data_list = data_list.get("data", [])
        if not isinstance(data_list, list):
            return {}
        for item in data_list:
            rid = item.get("resourceId", "")
            if rid:
                resources[rid] = {
                    "name": item.get("name", ""),
                    "description": item.get("description", ""),
                    "access": item.get("access", 0),
                }
        return resources
    except APIError as e:
        print(f"    获取资源定义失败 ({model}): {e}")
        return {}
    except Exception as e:
        print(f"    获取资源定义异常 ({model}): {e}")
        return {}


def _build_enrichment_db():
    """从已有 sample 中建立 resource_id -> {name, description} 的全局字典

    用于给探测模式补充元数据。
    """
    db = {}
    if not os.path.isdir(SAMPLES_DIR):
        return db
    for fname in os.listdir(SAMPLES_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(SAMPLES_DIR, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
        for rid, info in data.get("resources", {}).items():
            name = info.get("name", "")
            desc = info.get("description", "")
            if (name or desc) and rid not in db:
                db[rid] = {"name": name, "description": desc}
    return db


def _enrich_probed_resources(resources, enrichment_db):
    """用已有 sample 的元数据补充探测模式获取的空 name/description"""
    for rid, info in resources.items():
        if not info.get("name") and not info.get("description"):
            if rid in enrichment_db:
                info["name"] = enrichment_db[rid]["name"]
                info["description"] = enrichment_db[rid]["description"]


def probe_resources(client, did):
    """用已知资源 ID 列表探测设备实际支持的资源

    对于 query.resource.info 返回空的老型号设备，
    通过直接查询资源值来发现设备支持哪些资源。
    """
    found = {}
    batch_size = 20
    for i in range(0, len(ALL_KNOWN_RESOURCE_IDS), batch_size):
        batch = ALL_KNOWN_RESOURCE_IDS[i:i + batch_size]
        try:
            result = client.query_resource_value(did, batch)
            if result:
                data_list = result if isinstance(result, list) else result.get("data", [])
                if isinstance(data_list, dict):
                    data_list = [data_list]
                for item in data_list:
                    if isinstance(item, dict):
                        rid = item.get("resourceId", "")
                        if rid:
                            found[rid] = {
                                "name": "",
                                "description": "",
                                "access": 0,
                                "value": item.get("value", ""),
                            }
                        for res in item.get("resources", []):
                            rid = res.get("resourceId", "")
                            if rid:
                                found[rid] = {
                                    "name": "",
                                    "description": "",
                                    "access": 0,
                                    "value": res.get("value", ""),
                                }
        except APIError as e:
            if e.code == 302:
                print(f"    设备不可达 (302)，跳过探测")
                return found
        except Exception:
            pass
        if i + batch_size < len(ALL_KNOWN_RESOURCE_IDS):
            time.sleep(2)
    return found


def fetch_resource_values(client, did, resource_ids):
    """获取设备当前资源值"""
    if not resource_ids:
        return {}
    values = {}
    batch_size = 20
    for i in range(0, len(resource_ids), batch_size):
        batch = resource_ids[i:i + batch_size]
        try:
            result = client.query_resource_value(did, batch)
            if result:
                data_list = result if isinstance(result, list) else result.get("data", [])
                if isinstance(data_list, dict):
                    data_list = [data_list]
                for item in data_list:
                    if isinstance(item, dict):
                        rid = item.get("resourceId", "")
                        if rid:
                            values[rid] = item.get("value", "")
                        for res in item.get("resources", []):
                            rid = res.get("resourceId", "")
                            if rid:
                                values[rid] = res.get("value", "")
        except Exception as e:
            print(f"    获取资源值异常 ({did}): {e}")
        if i + batch_size < len(resource_ids):
            time.sleep(2)
    return values


def main():
    parser = argparse.ArgumentParser(description="采集 Aqara 设备数据")
    parser.add_argument("--model", help="只采集指定 model 的设备")
    parser.add_argument("--force", action="store_true",
                        help="强制重新采集（覆盖已有 sample）")
    args = parser.parse_args()

    config = load_config()
    client = AqaraClient(
        app_id=config["app_id"],
        app_key=config["app_key"],
        key_id=config["key_id"],
        country_code=config.get("country_code", "CN"),
        access_token=config["access_token"],
        refresh_token_value=config.get("refresh_token", ""),
    )

    official_names = load_official_names()
    existing_samples = load_existing_samples()

    print("正在获取设备列表...")
    try:
        devices = client.query_all_devices()
    except APIError as e:
        if e.code == 108:
            print("Token 已过期，请先运行: python tools/auth.py --refresh")
        else:
            print(f"获取设备列表失败: {e}")
        sys.exit(1)

    print(f"共获取到 {len(devices)} 个设备")

    # 加载已有 sample 用于补充探测模式的元数据
    enrichment_db = _build_enrichment_db()

    # 按 model 分组，保留所有设备（用于在一个不可达时尝试另一个）
    # 跳过 matter 桥接设备（无 Aqara 资源体系）和非标 model
    SKIP_PREFIXES = ("aqara.matter.", "app.")
    all_devices_by_model = {}
    models_seen = {}
    skipped_matter = 0
    for dev in devices:
        model = dev.get("model", "")
        if not model:
            continue
        if any(model.startswith(p) for p in SKIP_PREFIXES):
            skipped_matter += 1
            continue
        all_devices_by_model.setdefault(model, []).append(dev)
        if args.model and model != args.model:
            continue
        if model not in models_seen:
            models_seen[model] = dev  # 取第一个作为默认

    if skipped_matter:
        print(f"已跳过 {skipped_matter} 个 Matter/虚拟设备")

    # 增量模式：跳过已有且有资源的 sample；重采空资源的
    if not args.force and not args.model:
        skipped = 0
        filtered = {}
        for model, dev in models_seen.items():
            if model in existing_samples and existing_samples[model]:
                skipped += 1
            else:
                filtered[model] = dev
        if skipped:
            print(f"增量模式：跳过 {skipped} 个已有资源的 sample")
        models_seen = filtered

    total = len(models_seen)
    if total == 0:
        print("没有需要采集的设备")
        return

    print(f"共 {total} 种设备型号需要采集\n")

    saved = 0
    for idx, (model, dev) in enumerate(sorted(models_seen.items()), 1):
        did = dev.get("did", "")
        dev_name = official_names.get(model, model)
        print(f"[{idx}/{total}] {model} ({dev_name})")

        # 步骤1：尝试 query.resource.info 获取资源定义
        time.sleep(2)
        res_info = fetch_resource_info(client, model)

        if res_info:
            print(f"    资源定义: {len(res_info)} 个")
            # 获取资源值
            resource_ids = list(res_info.keys())
            time.sleep(2)
            values = fetch_resource_values(client, did, resource_ids)

            resources = {}
            for rid, info in res_info.items():
                resources[rid] = {
                    "name": info.get("name", ""),
                    "description": info.get("description", ""),
                    "access": info.get("access", 0),
                    "value": values.get(rid, ""),
                }
        else:
            # 步骤2：Fallback - 用已知资源 ID 直接探测
            # 如果当前设备不可达，遍历同 model 的其他设备
            candidates = all_devices_by_model.get(model, [dev])
            resources = {}
            for ci, cdev in enumerate(candidates):
                cdid = cdev.get("did", "")
                if ci == 0:
                    print(f"    资源定义为空，使用探测模式...")
                else:
                    print(f"    尝试第 {ci+1}/{len(candidates)} 个设备 ({cdid})...")
                time.sleep(2)
                resources = probe_resources(client, cdid)
                if resources:
                    _enrich_probed_resources(resources, enrichment_db)
                    print(f"    探测到 {len(resources)} 个资源")
                    break
            if not resources:
                print(f"    所有 {len(candidates)} 个设备均无法获取资源")

        sample = {
            "model": model,
            "device_name": dev_name,
            "model_type": dev.get("modelType", 0),
            "resources": resources,
            "fetched_at": datetime.now().isoformat(),
        }

        filepath = save_sample(model, sample)
        saved += 1
        print(f"    已保存: {filepath} ({len(resources)} 个资源)")

    print(f"\n采集完成: 成功 {saved}/{total}")


if __name__ == "__main__":
    main()
