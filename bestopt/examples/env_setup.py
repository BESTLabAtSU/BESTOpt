from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.environment import BESTOptEnvironment

cm = ConfigurationManager("config_setup.json")
env = BESTOptEnvironment(cm.config)

system_state = env.reset()
