#!/usr/bin/env python3
"""验证 aiot_mapping.py 与 sample 数据的一致性

对于没有 sample 的设备，会自动查找同系列设备（相同 model 前缀）的 sample 进行模拟验证。

用法:
    python tools/validate_mapping.py                       # 完整验证
    python tools/validate_mapping.py --strict               # 严格模式
    python tools/validate_mapping.py --model lumi.switch.acn056  # 只验证指定 model

退出码:
    0 - 全部通过
    1 - 存在资源 ID 不匹配
"""

import argparse
import json
import os
import re
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TOOLS_DIR)
MAPPING_PATH = os.path.join(
    PROJECT_DIR,
    "custom_components", "aqara_bridge", "core", "aiot_mapping.py",
)
SAMPLES_DIR = os.path.join(TOOLS_DIR, "samples")


def load_samples():
    """加载所有 sample JSON 文件"""
    samples = {}
    if not os.path.isdir(SAMPLES_DIR):
        return samples
    for fname in os.listdir(SAMPLES_DIR):
        if not fname.endswith(".json"):
            continue
        filepath = os.path.join(SAMPLES_DIR, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = data.get("model", fname.replace(".json", ""))
        samples[model] = data
    return samples


def get_model_family(model):
    """提取 model 的系列前缀，用于查找同类设备

    lumi.switch.acn056 -> lumi.switch
    lumi.curtain.hagl04 -> lumi.curtain
    aqara.matter.4447_4099 -> aqara.matter
    """
    parts = model.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return model


def find_similar_sample(model, samples):
    """查找同系列设备的 sample 用于模拟验证

    优先选择资源数量最多的同系列 sample。
    """
    family = get_model_family(model)
    candidates = []
    for s_model, s_data in samples.items():
        if get_model_family(s_model) == family and s_data.get("resources"):
            candidates.append((s_model, s_data))
    if not candidates:
        return None, None
    # 选资源最多的
    candidates.sort(key=lambda x: len(x[1].get("resources", {})), reverse=True)
    return candidates[0]


def parse_mapping():
    """解析 aiot_mapping.py 提取 model -> 资源 ID 映射

    返回: {model: {"name": str, "resource_ids": set, "has_params": bool}}
    """
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    model_pattern = re.compile(
        r'"((?:lumi|aqara|x1\.lumi)\.[a-zA-Z0-9_.]+)"'
        r'\s*:\s*\["([^"]*)",\s*"([^"]*)"'
    )
    resource_pattern = re.compile(r'\("(\d+\.(?:\{\}|\d+)\.\d+)"')
    ch_count_pattern = re.compile(r'"ch_count":\s*(\d+|None)')
    ch_start_pattern = re.compile(r'"ch_start":\s*(\d+)')

    results = {}

    mapping_start = content.find("AIOT_DEVICE_MAPPING = [")
    if mapping_start < 0:
        print("ERROR: 找不到 AIOT_DEVICE_MAPPING")
        sys.exit(1)

    mapping_content = content[mapping_start:]

    # 分割为顶层 dict 块
    blocks = []
    depth = 0
    block_start = None
    in_list = False

    for i, ch in enumerate(mapping_content):
        if ch == '[' and not in_list:
            in_list = True
            continue
        if not in_list:
            continue
        if ch == '{':
            if depth == 0:
                block_start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and block_start is not None:
                blocks.append(mapping_content[block_start:i+1])
                block_start = None

    for block in blocks:
        models_in_block = model_pattern.findall(block)
        if not models_in_block:
            continue

        has_params = '"params"' in block
        is_empty_params = '"params": []' in block or '"params":[]' in block

        raw_resource_ids = resource_pattern.findall(block)

        expanded_ids = set()
        for rid in raw_resource_ids:
            if '{}' in rid:
                rid_pos = block.find(f'"{rid}"')
                if rid_pos < 0:
                    continue
                context = block[max(0, rid_pos-500):rid_pos+500]
                ch_match = ch_count_pattern.search(context)
                ch_start_match = ch_start_pattern.search(context)

                if ch_match and ch_match.group(1) != "None":
                    ch_count = int(ch_match.group(1))
                    ch_start = int(ch_start_match.group(1)) if ch_start_match else 1
                    for ch in range(ch_count):
                        expanded_ids.add(rid.replace('{}', str(ch_start + ch)))
                else:
                    expanded_ids.add(rid)
            else:
                expanded_ids.add(rid)

        for model_id, brand, name in models_in_block:
            if model_id == "params":
                continue
            results[model_id] = {
                "name": name,
                "resource_ids": expanded_ids,
                "has_params": has_params and not is_empty_params,
            }

    return results


def validate(mapping, samples, target_model=None, strict=False):
    """验证 mapping 与 samples 的一致性

    没有直接 sample 的设备，会用同系列设备的 sample 模拟验证。
    """
    errors = 0
    warnings = 0
    passed = 0
    simulated = 0
    no_sample = 0

    models = sorted(mapping.keys())
    if target_model:
        models = [m for m in models if m == target_model]

    for model in models:
        info = mapping[model]
        if not info["has_params"]:
            continue

        # 查找 sample：优先直接匹配，其次同系列模拟
        sample = samples.get(model)
        sim_model = None
        if not sample:
            sim_model, sample = find_similar_sample(model, samples)

        if not sample or not sample.get("resources"):
            no_sample += 1
            if strict:
                print(f"  [NO DATA] {model} ({info['name']})")
            continue

        sample_rids = set(sample.get("resources", {}).keys())
        mapping_rids = info["resource_ids"]

        # 检查 mapping 引用了 sample 中不存在的资源
        missing_in_sample = mapping_rids - sample_rids
        # 过滤掉系统资源（8.0.xxxx）和模板资源
        missing_real = {r for r in missing_in_sample
                        if not r.startswith("8.0.") and '{}' not in r}

        # 检查 sample 中有但 mapping 未用的资源
        unused = sample_rids - mapping_rids
        unused_interesting = {r for r in unused if not r.startswith("8.0.")}

        label = f"{model} ({info['name']})"
        if sim_model:
            label += f" [模拟: {sim_model}]"
            simulated += 1

        if missing_real:
            # 模拟验证时降级为警告（同系列设备资源可能有差异）
            if sim_model:
                warnings += 1
                print(f"  [SIM WARN] {label}")
                print(f"         同系列设备缺少以下资源（可能是型号差异）:")
            else:
                errors += 1
                print(f"  [FAIL] {label}")
                print(f"         mapping 引用了 sample 中不存在的资源:")
            for rid in sorted(missing_real):
                print(f"           - {rid}")
        else:
            passed += 1

        # 只在直接匹配时显示未使用的资源
        if unused_interesting and not sim_model:
            if not missing_real:
                print(f"  [WARN] {label}")
            print(f"         sample 中存在但 mapping 未使用的资源:")
            for rid in sorted(unused_interesting):
                res = sample["resources"].get(rid, {})
                desc = res.get("description", res.get("name", ""))
                print(f"           + {rid} ({desc})")
            warnings += 1

    print(f"\n{'='*60}")
    print(f"验证结果:")
    print(f"  通过: {passed}")
    print(f"  失败（资源不匹配）: {errors}")
    print(f"  警告: {warnings}")
    print(f"  模拟验证（同系列）: {simulated}")
    print(f"  无可用数据: {no_sample}")
    total_with_params = sum(1 for m in mapping.values() if m["has_params"])
    print(f"  总计（有 params 的 model）: {total_with_params}")

    return errors


def main():
    parser = argparse.ArgumentParser(description="验证 aiot_mapping.py")
    parser.add_argument("--model", help="只验证指定 model")
    parser.add_argument("--strict", action="store_true",
                        help="严格模式，无 sample 也报告")
    args = parser.parse_args()

    print("加载 sample 数据...")
    samples = load_samples()
    print(f"  已加载 {len(samples)} 个 sample\n")

    print("解析 aiot_mapping.py...")
    mapping = parse_mapping()
    print(f"  已解析 {len(mapping)} 个设备 model\n")

    print("开始验证...\n")
    errors = validate(mapping, samples, args.model, args.strict)

    sys.exit(1 if errors > 0 else 0)


if __name__ == "__main__":
    main()
