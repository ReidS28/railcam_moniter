import os
import logging
from datetime import datetime
from slack_sdk import WebClient

# Set up logging (optional - you can remove this in production)
logging.basicConfig(level=logging.INFO)

# Initialize Slack client
slack_token = "xoxb-8678201375584-9289783952051-BanAx2McG4hrWrgf3nCg0D1Z"
user_id = "U08KY5XBKJ4"
client = WebClient(token=slack_token)

def send_slack_photo(image_path, message="", user_id=None, channel=None):
    """
    Upload and send a photo to Slack using the official slack-sdk library
    
    Args:
        image_path (str): Path to the image file
        message (str): Optional message to accompany the photo
        user_id (str, optional): User ID for direct message
        channel (str, optional): Channel ID or name for channel message
    """
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        # If user_id is provided, we need to open a DM channel first
        if user_id:
            # Open a direct message channel
            dm_response = client.conversations_open(users=[user_id])
            if not dm_response["ok"]:
                print(f"❌ Failed to open DM channel: {dm_response.get('error')}")
                return False
            target = dm_response["channel"]["id"]
        elif channel:
            target = channel
        else:
            print("Error: Must specify either user_id or channel")
            return False
        
        # Upload file using the official Slack SDK
        response = client.files_upload_v2(
            file=image_path,
            filename=os.path.basename(image_path),
            title=f"Train Detection - {os.path.basename(image_path)}",
            initial_comment=message,
            channel=target
        )
        
        if response["ok"]:
            print(f"✅ Photo sent successfully to {target}")
            return True
        else:
            print(f"❌ Failed to send photo: {response.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error uploading photo: {e}")
        return False

def send_slack_message(message, user_id=None, channel=None):
    """
    Send a text message to Slack using the official slack-sdk library
    
    Args:
        message (str): The message to send
        user_id (str, optional): User ID for direct message
        channel (str, optional): Channel ID or name for channel message
    """
    # Determine the target (DM or channel)
    if user_id:
        target = user_id
    elif channel:
        target = channel
    else:
        print("Error: Must specify either user_id or channel")
        return False
    
    try:
        response = client.chat_postMessage(
            channel=target,
            text=message
        )
        
        if response["ok"]:
            print(f"✅ Message sent successfully to {target}")
            return True
        else:
            print(f"❌ Failed to send message: {response.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error sending message: {e}")
        return False

def handle_train(path):
    """
    Handle train detection and send Slack notification with photo, then delete the file
    
    Args:
        path (str): Path to the detection image file
    """
    print(f"🚂 Train detected at: {path}")
    
    # Extract filename from path for more informative message
    filename = os.path.basename(path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the message to accompany the photo
    message = f"🚂 Train Detected @ {timestamp}"
    
    # Try to send the photo with message
    success = send_slack_photo(path, message, user_id=user_id)
    
    if not success:
        print("📱 Trying fallback: sending text message only...")
        # Fallback: send just the text message
        fallback_message = f"{message}\n\n📸 Image location: {path}"
        success = send_slack_message(fallback_message, user_id=user_id)
        
        if success:
            print("✅ Train detection message sent to Slack (without photo)")
        else:
            print("❌ Failed to send any notification to Slack")
    else:
        print("✅ Train detection photo sent to Slack successfully")
    
    # Delete the image file after sending notification
    try:
        if os.path.exists(path):
            os.remove(path)
            print(f"🗑️ Deleted image file: {path}")
        else:
            print(f"⚠️ Image file not found for deletion: {path}")
    except OSError as e:
        print(f"❌ Failed to delete image file: {e}")
    except Exception as e:
        print(f"❌ Unexpected error deleting file: {e}")

# Test the function
if __name__ == "__main__":
    # Test auth first
    try:
        auth_test = client.auth_test()
        bot_user_id = auth_test["user_id"]
        print(f"✅ Bot authenticated successfully: {bot_user_id}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
    
    # Run the train detection handler
    handle_train("detections/train_delayed_1754084410.jpg")