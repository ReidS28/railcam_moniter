import os
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from typing import Optional, List, Dict, Any
import configparser

from utils import print_time

config = configparser.ConfigParser()
config.read("config.ini")


class Slack:

    def __init__(self):
        config = configparser.ConfigParser()
        config.read("config.ini")

        load_dotenv()
        self.SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
        self.CHANNEL_NAME = config.get("slack", "channel_name")

        self.client = WebClient(token=self.SLACK_BOT_TOKEN)
        self.CHANNEL_ID = self._get_channel_id_by_name(self.CHANNEL_NAME)

    def _get_channel_id_by_name(self, channel_name: str) -> Optional[str]:
        clean_name = channel_name.lstrip("#")

        try:
            for page in self.client.conversations_list(
                types="public_channel,private_channel"
            ):
                for channel in page["channels"]:
                    if channel.get("name") == clean_name:
                        print_time(
                            f"Resolved channel name '{channel_name}' to ID: {channel['id']}"
                        )
                        return channel["id"]

            print_time(f"Error: Could not find channel named '{channel_name}'.", "red")
            return None

        except SlackApiError as e:
            error_message = e.response["error"]
            status_code = e.response.status_code
            print_time(
                f"Slack API Error resolving channel name: {error_message} (HTTP Status: {status_code})",
                "red",
            )
            return None

        except Exception as e:
            print_time(f"General Error fetching conversations: {e}", "red")
            return None

    def send_message(
        self, message: str, channel: str = None
    ) -> Optional[Dict[str, Any]]:

        target_channel = channel
        if target_channel is None:
            target_channel = self.CHANNEL_ID

        if not target_channel:
            print_time("Error: Channel ID is not resolved. Cannot send message.", "red")
            return None

        try:
            response = self.client.chat_postMessage(
                channel=target_channel, text=message
            )
            print_time(f"Message sent successfully to {channel}.", "green")
            return response
        except SlackApiError as e:
            print_time(
                f"Error sending message to {channel}: {e.response['error']} (HTTP Status: {e.response.status_code})",
                "red",
            )
            return None

    def upload_file(
        self, file_path: str, message: str, channel_id: str = None
    ) -> Optional[Dict[str, Any]]:

        target_channel_id = channel_id
        if target_channel_id is None:
            target_channel_id = self.CHANNEL_ID

        if not target_channel_id:
            print_time("FATAL: Channel ID is not resolved. Cannot send message.", "red")
            return None

        if not os.path.exists(file_path):
            print_time(f"FATAL: File not found at path: {file_path}", "red")
            return
        try:
            response = self.client.files_upload_v2(
                file=file_path,
                channel=target_channel_id,
                initial_comment=message,
            )
            print_time(
                f"File {os.path.basename(file_path)} uploaded successfully to {self.CHANNEL_NAME}."
            )
            return response
        except SlackApiError as e:
            print_time(
                f"Error uploading file: {e.response['error']} (HTTP Status: {e.response.status_code})",
                "red",
            )
            return None
        except FileNotFoundError:
            print_time(f"Error: File not found at path: {file_path}", "red")
            return None


if __name__ == "__main__":
    slack = Slack()

    message = "Test Message"

    slack.send_message(message)

    message = "Image"
    test_file_path = "image.png"
    # slack.upload_file(test_file_path, message)
