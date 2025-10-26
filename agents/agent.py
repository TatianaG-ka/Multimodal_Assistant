import logging

class Agent:
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    CYAN = "cyan"

    def log(self, msg: str):
        logging.getLogger().info(f"[{self.__class__.__name__}] {msg}")
