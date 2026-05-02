"""Aqara Cloud API 客户端（同步版）

复用 aiot_cloud.py 的签名逻辑，使用 requests 库进行同步请求。
支持自动重试和超时处理。
"""

import hashlib
import random
import string
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SERVER_URLS = {
    "CN": "https://open-cn.aqara.com/v3.0/open/api",
    "USA": "https://open-usa.aqara.com/v3.0/open/api",
    "KR": "https://open-kr.aqara.com/v3.0/open/api",
    "RU": "https://open-ru.aqara.com/v3.0/open/api",
    "GER": "https://open-ger.aqara.com/v3.0/open/api",
}


class AqaraClient:
    """Aqara Cloud API 同步客户端"""

    def __init__(self, app_id, app_key, key_id, country_code="CN",
                 access_token="", refresh_token_value=""):
        self.app_id = app_id
        self.app_key = app_key
        self.key_id = key_id
        self.api_url = SERVER_URLS.get(country_code, SERVER_URLS["CN"])
        self.access_token = access_token
        self.refresh_token_value = refresh_token_value
        self._session = self._make_session()

    @staticmethod
    def _make_session():
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=2,
                      status_forcelist=[502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        return session

    def _sign(self, nonce, timestamp):
        if self.access_token:
            s = (f"AccessToken={self.access_token}&Appid={self.app_id}"
                 f"&Keyid={self.key_id}&Nonce={nonce}&Time={timestamp}"
                 f"{self.app_key}")
        else:
            s = (f"Appid={self.app_id}&Keyid={self.key_id}"
                 f"&Nonce={nonce}&Time={timestamp}{self.app_key}")
        return hashlib.md5(s.lower().encode()).hexdigest()

    def _headers(self):
        nonce = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        timestamp = str(int(time.time() * 1000))
        sign = self._sign(nonce, timestamp)
        headers = {
            "Content-Type": "application/json",
            "Appid": self.app_id,
            "Keyid": self.key_id,
            "Nonce": nonce,
            "Time": timestamp,
            "Sign": sign,
            "Lang": "zh",
        }
        if self.access_token:
            headers["Accesstoken"] = self.access_token
        return headers

    def _request(self, intent, data=None, max_retries=2):
        payload = {"intent": intent, "data": data or {}}
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                resp = self._session.post(
                    self.api_url, headers=self._headers(),
                    json=payload, timeout=60,
                )
                resp.raise_for_status()
                result = resp.json()
                if result.get("code") != 0:
                    raise APIError(result.get("code"),
                                   result.get("message", "Unknown error"))
                return result.get("result")
            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as e:
                last_err = e
                if attempt < max_retries:
                    wait = 5 * (attempt + 1)
                    print(f"    网络超时，{wait}s 后重试 ({attempt+1}/{max_retries})...")
                    time.sleep(wait)
        raise last_err

    def get_auth_code(self, account, account_type=0):
        return self._request("config.auth.getAuthCode", {
            "account": account,
            "accountType": account_type,
            "accessTokenValidity": "7d",
        })

    def get_token(self, auth_code, account, account_type=0):
        result = self._request("config.auth.getToken", {
            "authCode": auth_code,
            "account": account,
            "accountType": account_type,
        })
        if result:
            self.access_token = result.get("accessToken", "")
            self.refresh_token_value = result.get("refreshToken", "")
        return result

    def refresh_token(self):
        if not self.refresh_token_value:
            raise ValueError("No refresh token available")
        result = self._request("config.auth.refreshToken", {
            "refreshToken": self.refresh_token_value,
        })
        if result:
            self.access_token = result.get("accessToken", "")
            self.refresh_token_value = result.get("refreshToken", "")
        return result

    def query_all_devices(self, page_size=50):
        """分页获取所有设备"""
        all_devices = []
        page = 1
        while True:
            result = self._request("query.device.info", {
                "pageNum": page,
                "pageSize": page_size,
            })
            data = result.get("data", []) if result else []
            all_devices.extend(data)
            if len(data) < page_size:
                break
            page += 1
            time.sleep(2)
        return all_devices

    def query_resource_info(self, model, resource_id=None):
        """查询设备 model 支持的资源定义"""
        data = {"model": model}
        if resource_id:
            data["resourceId"] = resource_id
        return self._request("query.resource.info", data)

    def query_resource_value(self, did, resource_ids):
        """查询设备当前资源值"""
        return self._request("query.resource.value", {
            "resources": [{
                "subjectId": did,
                "resourceIds": resource_ids,
            }]
        })


class APIError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(f"API Error {code}: {message}")
