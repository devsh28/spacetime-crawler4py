from configparser import ConfigParser
from argparse import ArgumentParser

from utils.server_registration import get_cache_server
from utils.config import Config
from crawler import Crawler
from scraper import write_report


def main(config_file, restart):
    try:
        # Create a ConfigParser instance and read the config file
        config_parser = ConfigParser()
        files_read = config_parser.read(config_file)
        if not files_read:
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
        
        # Check if the IDENTIFICATION section exists
        if "IDENTIFICATION" not in config_parser:
            raise KeyError("Missing [IDENTIFICATION] section in configuration file.")
        
        # Instantiate your Config using the loaded config_parser
        config = Config(config_parser)
        config.cache_server = get_cache_server(config, restart)
        
        # Start the crawler
        crawler = Crawler(config, restart)
        crawler.start()
        
    except Exception as e:
        print("Error in configuration:", e)
    finally:
        write_report()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--restart", action="store_true", default=False)
    parser.add_argument("--config_file", type=str, default="config.ini")
    args = parser.parse_args()
    main(args.config_file, args.restart)
