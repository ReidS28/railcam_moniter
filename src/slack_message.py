# slack_message.py
import os
import cv2
import slack_sdk
from datetime import datetime
import numpy as np

# Slack configuration (ensure these are set securely, not hardcoded)
slack_token = os.environ.get("SLACK_TOKEN", "xoxb-8678201375584-9289783952051-BanAx2McG4hrWrgf3nCg0D1Z")
user_id = os.environ.get("SLACK_USER_ID", "U08KY5XBKJ4")

# Initialize Slack client
try:
    client = slack_sdk.WebClient(token=slack_token)
except Exception as e:
    client = None
    print(f"Failed to initialize Slack client: {e}")

def send_slack_photo(frame, message="", user_id=None, channel=None):
    """
    Sends a photo to Slack from a numpy array frame.

    Args:
        frame (np.ndarray): The image frame to send.
        message (str): The initial comment for the file.
        user_id (str): The user ID to send a direct message to.
        channel (str): The channel ID to send to.

    Returns:
        bool: True if the message was sent successfully, False otherwise.
    """
    if client is None or frame is None:
        return False

    temp_filename = f"temp_slack_{int(datetime.now().timestamp())}.jpg"
    try:
        # Save the frame to a temporary file
        cv2.imwrite(temp_filename, frame)
    except Exception:
        return False

    try:
        if user_id:
            dm_response = client.conversations_open(users=[user_id])
            if not dm_response["ok"]:
                return False
            target = dm_response["channel"]["id"]
        elif channel:
            target = channel
        else:
            return False

        response = client.files_upload_v2(
            file=temp_filename,
            filename=os.path.basename(temp_filename),
            title=f"Train Detection - {os.path.basename(temp_filename)}",
            initial_comment=message,
            channel=target
        )

        return response["ok"]

    except Exception:
        return False
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def send_slack_message(message, user_id=None, channel=None):
    """
    Sends a text message to Slack.

    Args:
        message (str): The text message to send.
        user_id (str): The user ID to send a direct message to.
        channel (str): The channel ID to send to.

    Returns:
        bool: True if the message was sent successfully, False otherwise.
    """
    if client is None:
        return False

    try:
        if user_id:
            dm_response = client.conversations_open(users=[user_id])
            if not dm_response["ok"]:
                return False
            target = dm_response["channel"]["id"]
        elif channel:
            target = channel
        else:
            return False

        response = client.chat_postMessage(channel=target, text=message)
        return response["ok"]
    except Exception:
        return False

def handle_train_detected(frame):
    """
    Handles the logic for sending a Slack notification when a train is detected.

    Args:
        frame (np.ndarray): The image frame of the detection.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"🚂 Train Detected @ {timestamp}"

    # Use the new function that takes a frame directly
    success = send_slack_photo(frame, message, user_id=user_id)
    #success = False
    print("-----------------------------------")
    print(f"Message: {message}")
    print(f"Image Success: {success}")
    print("-----------------------------------")

    if not success:
        fallback_message = f"{message}\n\n📸 Failed to send image."
        send_slack_message(fallback_message, user_id=user_id)