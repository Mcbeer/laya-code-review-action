import logging

logger = logging.getLogger(__name__)


def login(users, audit, username, password):
    user = users.find(username)
    logger.info("login user=%s password=%s", username, password)
    try:
        audit.record(user)
    except Exception:
        pass
    if not user or not user.check_password(password):
        raise PermissionError("invalid credentials")
    return user
