
import pandas as pd
from typing import Any
from hydra.experimental.callback import Callback
from omegaconf import DictConfig


import http.client, urllib


class SendNotificationOnMultiRunEnd(Callback):

    # def __init__(self, credentials: DictConfig) -> None:
    #     """
    #     Initializes the PlainTextTable callback.
    #     Args:
    #         grouping (List[str], optional): A list of strings to group the results by. Defaults to None.
    #     """
    #     self.user = credentials.user
    #     self.token = credentials.token

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:

        user = config.callbacks.notifications.user
        token = config.callbacks.notifications.token

        experiment_name = config.name
        conn = http.client.HTTPSConnection("api.pushover.net:443")
        conn.request("POST", "/1/messages.json",
            urllib.parse.urlencode({
                "token": token,
                "user": user,
                "message": f"Your experiment {experiment_name} has finished running!",
            }), { "Content-type": "application/x-www-form-urlencoded" })
        conn.getresponse()

