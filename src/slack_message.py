import os
import cv2
from datetime import datetime
from utils import print_time, load_config
from slack_sdk.web.async_client import AsyncWebClient

CONFIG = load_config("config.json")

# Slack configuration
SLACK_TOKEN = CONFIG['general_settings']['slack_settings']['slack_token']
USER_ID = CONFIG['general_settings']['slack_settings']['slack_user_id']

# Initialize Slack client
try:
    client = AsyncWebClient(token=SLACK_TOKEN)
    print_time("Slack client initialized successfully", 'green')
except Exception as e:
    client = None
    print_time(f"Failed to initialize Slack client: {e}")

async def send_slack_photo(frame, message="", user_id=USER_ID, channel=None):
    if client is None or frame is None:
        print_time("Client not initialized or frame is None")
        return False
    
    temp_filename = f"temp_slack_{int(datetime.now().timestamp())}.jpg"
    
    try:
        # Save the frame as a temporary image file
        success = cv2.imwrite(temp_filename, frame)
        if not success:
            print_time("Failed to save image file")
            return False
    except Exception as e:
        print_time(f"Error saving image: {e}")
        return False
    
    try:
        # Determine target channel
        target = channel
        if user_id and not channel:
            # Open DM with user
            dm_response = await client.conversations_open(users=[user_id])
            if not dm_response["ok"]:
                print_time(f"Failed to open DM: {dm_response.get('error', 'Unknown error')}")
                return False
            target = dm_response["channel"]["id"]
        elif not channel:
            print_time("No channel or user_id specified")
            return False
        
        upload_response = await client.files_getUploadURLExternal(
            filename=os.path.basename(temp_filename),
            length=os.path.getsize(temp_filename)
        )
        
        if not upload_response["ok"]:
            print_time(f"Failed to get upload URL: {upload_response.get('error', 'Unknown error')}")
            return False
        
        upload_url = upload_response["upload_url"]
        file_id = upload_response["file_id"]
        
        import aiohttp
        async with aiohttp.ClientSession() as session:
            with open(temp_filename, 'rb') as file_data:
                async with session.post(upload_url, data=file_data) as upload_result:
                    if upload_result.status != 200:
                        print_time(f"File upload failed with status: {upload_result.status}", 'red')
                        return False

        complete_response = await client.files_completeUploadExternal(
            files=[{
                "id": file_id,
                "title": f"Train Detection - {os.path.basename(temp_filename)}"
            }],
            channel_id=target,
            initial_comment=message if message else None
        )
        
        if complete_response["ok"]:
            return True
        else:
            print_time(f"Failed to complete upload: {complete_response.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print_time(f"Error uploading to Slack: {e}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception as e:
                print_time(f"Failed to remove temp file: {e}")

async def send_slack_video(video_path, message="", user_id=USER_ID, channel=None):
    """
    Asynchronously sends a video file to Slack.
    """
    if client is None:
        print_time("Slack client not initialized")
        return False
    
    if not os.path.exists(video_path):
        print_time(f"Error: Video file not found at {video_path}")
        return False

    try:
        # Determine target channel
        target = channel
        if user_id and not channel:
            dm_response = await client.conversations_open(users=[user_id])
            if not dm_response["ok"]:
                print_time(f"Failed to open DM: {dm_response.get('error', 'Unknown error')}")
                return False
            target = dm_response["channel"]["id"]
        elif not channel:
            print_time("No channel or user_id specified")
            return False

        # Get upload URL from Slack
        upload_response = await client.files_getUploadURLExternal(
            filename=os.path.basename(video_path),
            length=os.path.getsize(video_path)
        )
        
        if not upload_response["ok"]:
            print_time(f"Failed to get upload URL: {upload_response.get('error', 'Unknown error')}")
            return False
        
        upload_url = upload_response["upload_url"]
        file_id = upload_response["file_id"]

        # Upload the video file using aiohttp
        import aiohttp
        async with aiohttp.ClientSession() as session:
            with open(video_path, 'rb') as file_data:
                async with session.post(upload_url, data=file_data) as upload_result:
                    if upload_result.status != 200:
                        print_time(f"File upload failed with status: {upload_result.status}", 'red')
                        return False

        # Complete the upload and send the message
        complete_response = await client.files_completeUploadExternal(
            files=[{
                "id": file_id,
                "title": f"Train Detection Video - {os.path.basename(video_path)}"
            }],
            channel_id=target,
            initial_comment=message if message else None
        )
        
        if complete_response["ok"]:
            print_time(f"Video '{os.path.basename(video_path)}' sent successfully to Slack")
            return True
        else:
            print_time(f"Failed to complete upload: {complete_response.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print_time(f"Error uploading video to Slack: {e}")
        return False

async def send_slack_message(message, user_id=USER_ID, channel=None):
    """
    Asynchronously sends a text message to Slack.
    """
    if client is None:
        print_time("Slack client not initialized")
        return False
    
    try:
        target = channel
        if user_id and not channel:
            dm_response = await client.conversations_open(users=[user_id])
            if not dm_response["ok"]:
                print_time(f"Failed to open DM: {dm_response.get('error', 'Unknown error')}")
                return False
            target = dm_response["channel"]["id"]
        elif not channel:
            print_time("No channel or user_id specified")
            return False
        
        response = await client.chat_postMessage(
            channel=target, 
            text=message
        )
        
        if response["ok"]:
            print_time("Message sent successfully to Slack")
            return True
        else:
            print_time(f"Failed to send message: {response.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print_time(f"Error sending message to Slack: {e}")
        return False