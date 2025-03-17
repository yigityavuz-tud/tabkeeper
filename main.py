import yaml
from tabkeeper import process_data

# Read the config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Run the data processing function
process_data(config)