import os
import shelve

from threading import Thread, RLock
from queue import Queue, Empty

from utils import get_logger, get_urlhash, normalize
from scraper import is_valid

class Frontier(object):
    def __init__(self, config, restart):
        self.logger = get_logger("FRONTIER")
        self.config = config
        self.to_be_downloaded = list()
        self.restart = restart  # Store restart status

        #clean shelve files when restart
        self.cleanup_shelve_files()
        
        if not os.path.exists(self.config.save_file) and not restart:
            # Save file does not exist, but request to load save.
            self.logger.info(
                f"Did not find save file {self.config.save_file}, "
                f"starting from seed.")
        elif os.path.exists(self.config.save_file) and restart:
            # Save file does exists, but request to start from seed.
            self.logger.info(
                f"Found save file {self.config.save_file}, deleting it.")
            os.remove(self.config.save_file)
        # Load existing save file, or create one if it does not exist.
        self.save = shelve.open(self.config.save_file)
        if restart:
            for url in self.config.seed_urls:
                self.add_url(url)
        else:
            # Set the frontier state with contents of save file.
            self._parse_save_file()
            if not self.save:
                for url in self.config.seed_urls:
                    self.add_url(url)

    def cleanup_shelve_files(self):

        if self.restart:
            shelve_files = [
                'words.shelve',
                'dupe_cache.shelve',
                'subdomains.shelve',
                'cache.shelve'
            ]
            for file in shelve_files:
                if os.path.exists(file):
                    try:
                        os.remove(file)
                        self.logger.info(f"Deleted {file}")
                    except Exception as e:
                        self.logger.error(f"Error deleting {file}: {e}")
                        
            # Also remove any .db, .dir, .dat files associated with shelves
            for file in shelve_files:
                for ext in ['.db', '.dir', '.dat']:
                    full_path = file + ext
                    if os.path.exists(full_path):
                        try:
                            os.remove(full_path)
                            self.logger.info(f"Deleted {full_path}")
                        except Exception as e:
                            self.logger.error(f"Error deleting {full_path}: {e}")

    def _parse_save_file(self):
        ''' This function can be overridden for alternate saving techniques. '''
        total_count = len(self.save)
        tbd_count = 0
        for url, completed in self.save.values():
            if not completed and is_valid(url):
                self.to_be_downloaded.append(url)
                tbd_count += 1
        self.logger.info(
            f"Found {tbd_count} urls to be downloaded from {total_count} "
            f"total urls discovered.")

    def get_tbd_url(self):
        try:
            return self.to_be_downloaded.pop()
        except IndexError:
            return None

    def add_url(self, url):
        try:
            url = normalize(url)
            urlhash = get_urlhash(url)
            if urlhash not in self.save:
                self.save[urlhash] = (url, False)
                self.save.sync()
                self.to_be_downloaded.append(url)
        except Exception as e:
            self.logger.error(f"Error adding URL {url}: {e}")
    
    def mark_url_complete(self, url):
        try:
            urlhash = get_urlhash(url)
            if urlhash not in self.save:
                # This should not happen.
                self.logger.error(
                    f"Completed url {url}, but have not seen it before.")
            else:
                self.save[urlhash] = (url, True)
                self.save.sync()
        except Exception as e:
            self.logger.error(f"Error marking URL complete {url}: {e}")

    def __del__(self):
        """
        Cleanup when the frontier is destroyed.
        """
        try:
            self.save.close()
        except Exception as e:
            self.logger.error(f"Error closing save file: {e}")