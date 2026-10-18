import configparser

config = configparser.ConfigParser()
config.read("config.ini")

active_profile = config.get("general", "active_profile")
if config.has_section(active_profile):
    profile = config[active_profile]
else:
    raise ValueError(f"Profile '{active_profile}' defined in [general] does not exist as a section in config.ini!")