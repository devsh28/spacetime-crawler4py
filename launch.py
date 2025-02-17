from configparser import ConfigParser
from argparse import ArgumentParser

from utils.server_registration import get_cache_server
from utils.config import Config
from crawler import Crawler


def main(config_file, restart):
    try:
        config = Config(ConfigParser())
        config.cache_server = get_cache_server(config, restart)
        crawler = Crawler(config, restart)
        crawler.start()
    finally:
        from scraper import write_report
        write_report()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--restart", action="store_true", default=False)
    parser.add_argument("--config_file", type=str, default="config.ini")
    args = parser.parse_args()
    main(args.config_file, args.restart)
