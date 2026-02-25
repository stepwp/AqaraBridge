import datetime
from email import message
import re
import logging
from datetime import timedelta

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers import aiohttp_client
from homeassistant.helpers.event import async_track_time_interval

from .core.aiot_manager import (
    AiotManager,
    AiotDevice,
)
from .core.aiot_cloud import AiotCloud
from .core.const import *


_LOGGER = logging.getLogger(__name__)

_DEBUG_ACCESSTOKEN = ""
_DEBUG_REFRESHTOEEN = ""
_DEBUG_STATUS = False


def data_masking(s: str, n: int) -> str:
    return re.sub(f"(?<=.{{{n}}}).(?=.{{{n}}})", "*", str(s))


def gen_auth_entry(
    app_id: str,
    app_key: str,
    key_id: str,
    account: str,
    account_type: int,
    country_code: str,
    token_result: dict,
):
    auth_entry = {}
    auth_entry[CONF_ENTRY_APP_ID] = app_id
    auth_entry[CONF_ENTRY_APP_KEY] = app_key
    auth_entry[CONF_ENTRY_KEY_ID] = key_id
    auth_entry[CONF_ENTRY_AUTH_ACCOUNT] = account
    auth_entry[CONF_ENTRY_AUTH_ACCOUNT_TYPE] = account_type
    auth_entry[CONF_ENTRY_AUTH_COUNTRY_CODE] = country_code
    auth_entry[CONF_ENTRY_AUTH_OPENID] = token_result["openId"]
    auth_entry[CONF_ENTRY_AUTH_ACCESS_TOKEN] = token_result["accessToken"]
    auth_entry[CONF_ENTRY_AUTH_EXPIRES_IN] = token_result["expiresIn"]
    auth_entry[CONF_ENTRY_AUTH_EXPIRES_TIME] = (
        datetime.datetime.now()
        + datetime.timedelta(seconds=int(token_result["expiresIn"]))
    ).strftime("%Y-%m-%d %H:%M:%S")
    auth_entry[CONF_ENTRY_AUTH_REFRESH_TOKEN] = token_result["refreshToken"]
    return auth_entry


def init_hass_data(hass):
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN].setdefault(HASS_DATA_AUTH_ENTRY_ID, None)
    session = AiotCloud(aiohttp_client.async_create_clientsession(hass))
    if not hass.data[DOMAIN].get(HASS_DATA_AIOTCLOUD):
        hass.data[DOMAIN].setdefault(HASS_DATA_AIOTCLOUD, session)
    if not hass.data[DOMAIN].get(HASS_DATA_AIOT_MANAGER):
        hass.data[DOMAIN].setdefault(HASS_DATA_AIOT_MANAGER, AiotManager(hass, session))


async def async_setup(hass, config):
    """Setup component."""
    init_hass_data(hass)
    return True


async def async_refresh_token_handler(hass: HomeAssistant, entry: ConfigEntry):
    """定期检查并刷新令牌的处理函数"""
    try:
        data = entry.data.copy()
        aiotcloud: AiotCloud = hass.data[DOMAIN][HASS_DATA_AIOTCLOUD]

        # 解析过期时间
        expires_time = datetime.datetime.strptime(
            data.get(CONF_ENTRY_AUTH_EXPIRES_TIME), "%Y-%m-%d %H:%M:%S"
        )

        # 计算剩余时间
        now = datetime.datetime.now()
        time_remaining = expires_time - now

        # 如果令牌在指定天数内即将过期，执行刷新
        if time_remaining <= timedelta(days=TOKEN_REFRESH_ADVANCE_DAYS):
            _LOGGER.info(
                f"Token will expire in {time_remaining}, refreshing now..."
            )

            # 执行令牌刷新
            resp = await aiotcloud.async_refresh_token(
                data.get(CONF_ENTRY_AUTH_REFRESH_TOKEN)
            )

            if isinstance(resp, dict) and resp.get("code") == 0:
                # 刷新成功，更新配置
                auth_entry = gen_auth_entry(
                    data.get(CONF_ENTRY_APP_ID),
                    data.get(CONF_ENTRY_APP_KEY),
                    data.get(CONF_ENTRY_KEY_ID),
                    data.get(CONF_ENTRY_AUTH_ACCOUNT),
                    data.get(CONF_ENTRY_AUTH_ACCOUNT_TYPE),
                    data.get(CONF_ENTRY_AUTH_COUNTRY_CODE),
                    resp["result"],
                )
                hass.config_entries.async_update_entry(entry, data=auth_entry)

                # 更新 aiotcloud 中的令牌
                aiotcloud.access_token = resp["result"]["accessToken"]
                aiotcloud.refresh_token = resp["result"]["refreshToken"]

                _LOGGER.info("Token refreshed successfully")
            else:
                # 刷新失败，记录错误并降级运行
                _LOGGER.error(
                    f"Token refresh failed: {resp}. Running in degraded mode."
                )
        else:
            _LOGGER.debug(
                f"Token is still valid, expires in {time_remaining}"
            )

    except Exception as e:
        _LOGGER.error(f"Error in token refresh handler: {e}", exc_info=True)


async def async_setup_token_refresh(hass: HomeAssistant, entry: ConfigEntry):
    """设置令牌自动刷新定时器"""
    # 立即检查一次令牌状态
    await async_refresh_token_handler(hass, entry)

    # 设置定期检查（每小时检查一次）
    async def _refresh_wrapper(now):
        await async_refresh_token_handler(hass, entry)

    cancel_timer = async_track_time_interval(
        hass,
        _refresh_wrapper,
        timedelta(hours=TOKEN_CHECK_INTERVAL_HOURS)
    )

    # 保存取消函数，以便后续清理
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}
    hass.data[DOMAIN][HASS_DATA_TOKEN_REFRESH_TIMER] = cancel_timer

    _LOGGER.info(
        f"Token refresh timer started, checking every {TOKEN_CHECK_INTERVAL_HOURS} hour(s)"
    )


async def async_cancel_token_refresh(hass: HomeAssistant):
    """取消令牌刷新定时器"""
    if DOMAIN in hass.data:
        cancel_timer = hass.data[DOMAIN].get(HASS_DATA_TOKEN_REFRESH_TIMER)
        if cancel_timer:
            cancel_timer()
            hass.data[DOMAIN][HASS_DATA_TOKEN_REFRESH_TIMER] = None
            _LOGGER.info("Token refresh timer cancelled")


async def async_setup_entry(hass, entry):
    def token_updated(access_token, refresh_token):
        auth_entry = hass.data[DOMAIN][HASS_DATA_AUTH_ENTRY_ID]
        if auth_entry:
            data = auth_entry.data.copy()
            data[CONF_ENTRY_AUTH_ACCESS_TOKEN] = access_token
            data[CONF_ENTRY_AUTH_REFRESH_TOKEN] = refresh_token
            hass.config_entries.async_update_entry(entry, data=data)

    # add update handler
    if not entry.update_listeners:
        entry.add_update_listener(async_update_options)

    data = entry.data.copy()
    if _DEBUG_STATUS:
        import time

        data[CONF_ENTRY_AUTH_REFRESH_TOKEN] = _DEBUG_REFRESHTOEEN
        data[CONF_ENTRY_AUTH_ACCESS_TOKEN] = _DEBUG_ACCESSTOKEN
        data[CONF_ENTRY_AUTH_EXPIRES_TIME] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(time.time() + 24 * 3600)
        )

    manager: AiotManager = hass.data[DOMAIN][HASS_DATA_AIOT_MANAGER]
    aiotcloud: AiotCloud = hass.data[DOMAIN][HASS_DATA_AIOTCLOUD]
    aiotcloud.set_options(entry.options)
    aiotcloud.set_app_id(data[CONF_ENTRY_APP_ID])
    aiotcloud.set_app_key(data[CONF_ENTRY_APP_KEY])
    aiotcloud.set_key_id(data[CONF_ENTRY_KEY_ID])
    aiotcloud.update_token_event_callback = token_updated
    if manager._msg_handler is not None:
        # 如果重新配置，重新启动mq
        manager._msg_handler.stop()
    await manager.start_msg_hanlder(
        data[CONF_ENTRY_APP_ID], data[CONF_ENTRY_APP_KEY], data[CONF_ENTRY_KEY_ID]
    )
    if (
        datetime.datetime.strptime(
            data.get(CONF_ENTRY_AUTH_EXPIRES_TIME), "%Y-%m-%d %H:%M:%S"
        )
        <= datetime.datetime.now()
    ):
        resp = aiotcloud.async_refresh_token(data.get(CONF_ENTRY_AUTH_REFRESH_TOKEN))
        if isinstance(resp, dict) and resp["code"] == 0:
            auth_entry = gen_auth_entry(
                data.get(CONF_ENTRY_AUTH_ACCOUNT),
                data.get(CONF_ENTRY_AUTH_ACCOUNT_TYPE),
                data.get(CONF_ENTRY_AUTH_COUNTRY_CODE),
                resp["result"],
            )
            hass.config_entries.async_update_entry(entry, data=auth_entry)
        else:
            # TODO 这里需要处理刷新令牌失败的情况
            return False
    else:
        aiotcloud.set_country(data.get(CONF_ENTRY_AUTH_COUNTRY_CODE))
        aiotcloud.access_token = data.get(CONF_ENTRY_AUTH_ACCESS_TOKEN)
        aiotcloud.refresh_token = data.get(CONF_ENTRY_AUTH_REFRESH_TOKEN)

    hass.data[DOMAIN][HASS_DATA_AUTH_ENTRY_ID] = entry

    if len(manager.all_devices) == 0:
        await manager.async_add_all_devices(entry)
        await manager.async_forward_entry_setup(entry)
    else:
        await manager.async_add_all_devices(entry)

    # 启动令牌自动刷新定时器
    await async_setup_token_refresh(hass, entry)

    return True


async def async_unload_entry(hass, entry):
    # 取消令牌刷新定时器
    await async_cancel_token_refresh(hass)

    # if CONF_ENTRY_AUTH_ACCOUNT in entry.data:
    #     hass.data[DOMAIN][HASS_DATA_AUTH_ENTRY_ID] = None
    # else:
    #     manager: AiotManager = hass.data[DOMAIN][HASS_DATA_AIOT_MANAGER]
    #     await manager.async_unload_entry(entry)
    return True


async def async_remove_entry(hass, entry):
    if CONF_ENTRY_AUTH_ACCOUNT in entry.data:
        hass.data[DOMAIN][HASS_DATA_AUTH_ENTRY_ID] = None
    else:
        manager: AiotManager = hass.data[DOMAIN][HASS_DATA_AIOT_MANAGER]
        await manager.async_remove_entry(entry)
    return True


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry):
    """Update Optioins if available"""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_remove_config_entry_device(
    hass: HomeAssistant, config_entry: ConfigEntry, device_entry
) -> bool:
    """Remove a config entry from a device."""
    return True
