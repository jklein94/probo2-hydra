
import pandas as pd
from typing import Any
from hydra.experimental.callback import Callback
from omegaconf import DictConfig


import http.client, urllib


class SendNotificationOnMultiRunEnd(Callback):
    """
    A custom callback that sends a notification when a multi-run experiment ends.

    This callback uses the Pushover API to notify the user when the experiment has completed.

    Attributes:
        None
    """

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Sends a Pushover notification when the multi-run experiment finishes.

        Args:
            config (DictConfig): The configuration object containing the experiment details.
                - `config.callbacks.notifications.user` (str): The Pushover user key.
                - `config.callbacks.notifications.token` (str): The Pushover API token.
                - `config.name` (str): The name of the experiment.
            **kwargs (Any): Additional keyword arguments (not used in this implementation).

        Returns:
            None

        Example:
            Given a `config` object with the following structure:
            ```
            config:
                name: "MyExperiment"
                callbacks:
                    notifications:
                        user: "user_key"
                        token: "api_token"
            ```
            This method sends a Pushover notification to the user stating:
            "Your experiment MyExperiment has finished running!"
        """
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


