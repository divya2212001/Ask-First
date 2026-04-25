# core/loader.py
# REPLACE YOUR OLD FILE WITH THIS

import json


def load_dataset(uploaded_file):
    """
    Supports Ask First official dataset format.

    Root:
    {
      "users": [...]
    }

    Each user contains:
    - conversations   (official dataset)
    or
    - sessions        (fallback older format)
    """

    try:
        if isinstance(uploaded_file, str):
            with open(uploaded_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.load(uploaded_file)

        validate_dataset(data)

        return data

    except Exception as e:
        raise ValueError(f"Failed to load dataset: {str(e)}")


def validate_dataset(data):

    if not isinstance(data, dict):
        raise ValueError("Dataset must be JSON object.")

    if "users" not in data:
        raise ValueError("Missing 'users' key.")

    if not isinstance(data["users"], list):
        raise ValueError("'users' must be a list.")

    for user in data["users"]:

        if "user_id" not in user:
            raise ValueError("Each user needs user_id")

        # official company format uses conversations
        if "conversations" not in user and "sessions" not in user:
            raise ValueError(
                f"User {user['user_id']} missing conversations"
            )

        # normalize automatically
        if "sessions" not in user and "conversations" in user:
            user["sessions"] = user["conversations"]