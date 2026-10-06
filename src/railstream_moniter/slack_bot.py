import logging
import os

from dotenv import find_dotenv, load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from utils import print_time

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)


class SlackBot:
    def __init__(self, channel: str) -> None:
        load_dotenv(find_dotenv())

        slack_token = os.environ.get("SLACK_BOT_TOKEN")
        self.client = WebClient(token=slack_token, timeout=300)

        self.channel = self._resolve_channel_id(channel)

    def _resolve_channel_id(self, channel: str) -> str:
        clean_channel = channel.lstrip("#")

        if clean_channel.startswith(("C", "G", "D")) and clean_channel.isupper():
            return clean_channel

        try:
            cursor = None
            while True:
                response = self.client.conversations_list(
                    types="public_channel,private_channel",
                    cursor=cursor,
                    limit=200,
                )

                for ch in response.get("channels", []):
                    if ch["name"] == clean_channel or ch["id"] == clean_channel:
                        return ch["id"]

                cursor = response.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

        except SlackApiError as e:
            print_time(f"Error resolving channel name '{channel}': {e}", "red")

        return clean_channel

    def send_message(self, message: str | None = None, file_path: str | None = None):
        if not message and not file_path:
            print_time(
                "Neither message nor file_path was provided. Nothing to send.", "yellow"
            )
            return None

        if file_path:
            try:
                print_time(f"Sending message with file: {file_path}")
                response = self.client.files_upload_v2(
                    channel=self.channel,
                    file=file_path,
                    initial_comment=message,
                )
                return response
            except (SlackApiError, OSError, TimeoutError) as e:
                print_time(
                    f"Failed to upload file '{file_path}': {e}. Falling back to text message...",
                    "red",
                )

        if message:
            try:
                print_time("Sending text message...")
                response = self.client.chat_postMessage(
                    channel=self.channel,
                    text=message,
                )
                return response
            except SlackApiError as e:
                print_time(f"Failed to send text message: {e}", "red")
        else:
            print_time(
                "File upload failed and no text message was provided for fallback.",
                "yellow",
            )

        return None
