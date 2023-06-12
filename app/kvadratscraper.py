import os
import pandas as pd
from typing import List, Dict, Any
import json
from models import ConsultantProfile
from webscraper import WebScraper
from dotenv import load_dotenv

load_dotenv()


class KvadratScraper:
    def _scrape_data(self, pages: int):
        scraper = WebScraper()
        for i in range(1, pages):
            starting_url = f'https://www.kvadrat.se/anlita-kvadrat/hitta-konsult/?q=&t=&s=&l=&p={i}#results'
            scraper.scrape_consultant_profile_pages(starting_url)
        return scraper.consultant_profiles


    def _save_data_csv(self, filename: str, consultant_profiles: ConsultantProfile):
        # new pandas dataframe
        total_df = pd.DataFrame(columns=['name',
                                        'title',
                                        'preamble',
                                        'article',
                                        'competence_list',
                                        'cv_list',
                                        'employment_list',
                                        'education_list'])

        serialized ='['
        for profile in consultant_profiles:
            profile.name = profile.name.strip()
            profile.title = profile.title.strip()
            profile.preamble = profile.preamble.strip()
            profile.article = profile.article.strip()

            # append all competences to a string
            json_data = json.dumps(profile.__dict__, ensure_ascii=False).encode('utf8').decode()
            if profile == consultant_profiles[-1]:
                serialized += json_data
            else:
                serialized += json_data + ','

        serialized += ']'

        pdf = pd.read_json(serialized)
        pdf.to_json(filename, orient='table', force_ascii=False)

    def download_profiles(self, filename: str, pages: int):
        consultant_profiles = self._scrape_data(pages)
        self._save_data_csv(filename, consultant_profiles)


if __name__ == '__main__':
    profile_location = os.environ['KGPT_PROFILE_LOCATION']
    pages = int(os.environ['KGPT_PAGES_TO_SCRAPE'])

    kvadratscraper = KvadratScraper()
    kvadratscraper.download_profiles(profile_location, pages=pages)