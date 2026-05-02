#!/usr/bin/env python3
"""Aqara API 交互式登录工具

用法:
    python tools/auth.py              # 交互式登录或刷新 token
    python tools/auth.py --refresh    # 强制刷新 token
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import yaml

from aqara_client import AqaraClient, APIError

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"配置文件不存在: {CONFIG_PATH}")
        print(f"请先复制模板: cp tools/config.example.yaml tools/config.yaml")
        sys.exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"配置已保存到 {CONFIG_PATH}")


def make_client(config):
    return AqaraClient(
        app_id=config["app_id"],
        app_key=config["app_key"],
        key_id=config["key_id"],
        country_code=config.get("country_code", "CN"),
        access_token=config.get("access_token", ""),
        refresh_token_value=config.get("refresh_token", ""),
    )


def is_token_valid(config):
    expires = config.get("expires_time", "")
    if not expires or not config.get("access_token"):
        return False
    try:
        exp_dt = datetime.fromisoformat(expires)
        return datetime.now() < exp_dt
    except (ValueError, TypeError):
        return False


def do_refresh(client, config):
    print("正在刷新 token...")
    try:
        result = client.refresh_token()
        config["access_token"] = result["accessToken"]
        config["refresh_token"] = result["refreshToken"]
        expires_in = int(result.get("expiresIn", 7776000))
        config["expires_time"] = (
            datetime.now() + timedelta(seconds=expires_in)
        ).isoformat()
        save_config(config)
        print("Token 刷新成功!")
        print(f"  有效期至: {config['expires_time']}")
        return True
    except APIError as e:
        print(f"刷新失败: {e}")
        return False


def do_login(client, config):
    account = config["account"]
    print(f"正在向 {account} 发送验证码...")
    try:
        client.get_auth_code(account)
        print("验证码已发送，请查收短信或邮件。")
    except APIError as e:
        print(f"发送验证码失败: {e}")
        sys.exit(1)

    auth_code = input("请输入验证码: ").strip()
    if not auth_code:
        print("验证码不能为空")
        sys.exit(1)

    try:
        result = client.get_token(auth_code, account)
        config["access_token"] = result["accessToken"]
        config["refresh_token"] = result["refreshToken"]
        expires_in = int(result.get("expiresIn", 7776000))
        config["expires_time"] = (
            datetime.now() + timedelta(seconds=expires_in)
        ).isoformat()
        save_config(config)
        print("登录成功!")
        print(f"  OpenID: {result.get('openId', 'N/A')}")
        print(f"  有效期至: {config['expires_time']}")
    except APIError as e:
        print(f"登录失败: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Aqara API 登录工具")
    parser.add_argument("--refresh", action="store_true", help="强制刷新 token")
    args = parser.parse_args()

    config = load_config()

    for key in ("app_id", "app_key", "key_id", "account"):
        if not config.get(key) or config[key].startswith("your_"):
            print(f"请在 {CONFIG_PATH} 中填写 {key}")
            sys.exit(1)

    client = make_client(config)

    if args.refresh:
        if config.get("refresh_token"):
            if not do_refresh(client, config):
                print("刷新失败，尝试重新登录...")
                do_login(client, config)
        else:
            print("没有 refresh_token，需要重新登录")
            do_login(client, config)
        return

    if is_token_valid(config):
        print(f"当前 token 有效，到期时间: {config['expires_time']}")
        choice = input("是否刷新 token? [y/N]: ").strip().lower()
        if choice == "y":
            do_refresh(client, config)
        return

    if config.get("refresh_token"):
        print("Token 已过期，尝试使用 refresh_token 刷新...")
        if do_refresh(client, config):
            return
        print("刷新失败，需要重新登录")

    do_login(client, config)


if __name__ == "__main__":
    main()
