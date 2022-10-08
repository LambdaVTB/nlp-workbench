import feedparser

# SOURCES = {
#     # Habr
#     # 'habr': 'https://habr.com/ru/rss/all/all/?fl=ru',

#     # Rubase
#     ## By companies
#     # 'rb_chance': "https://rb.ru/feeds/tag/chance/",
#     # 'rb_vk': "https://rb.ru/feeds/tag/vk/",
#     'rb_rvc': "https://rb.ru/feeds/tag/rvc/",
#     # 'rb_yandex': "https://rb.ru/feeds/tag/yandex/",
#     'rb_skolkovo': "https://rb.ru/feeds/tag/skolkovo/",
#     # 'rb_facebook': "https://rb.ru/feeds/tag/facebook/",
#     'rb_mailru': "https://rb.ru/feeds/tag/mailru/",
#     # 'rb_microsoft': "https://rb.ru/feeds/tag/microsoft/",

#     ## By topics
#     'rb_advertising': "https://rb.ru/feeds/tag/advertising/",
#     # 'rb_robotics': "https://rb.ru/feeds/tag/robotics/",
#     # 'rb_it': "https://rb.ru/feeds/tag/it/",
#     # 'rb_bigdata': "https://rb.ru/feeds/tag/bigdata/",
#     'rb_china': "https://rb.ru/feeds/tag/china/",
#     'rb_finance': "https://rb.ru/feeds/tag/fintech/",
#     # 'rb_cloud': "https://rb.ru/feeds/tag/cloud/",

#     # Vedomosti
#     # 'vd_business': "https://www.vedomosti.ru/rss/rubric/business",
#     # 'vd_it_business': "https://www.vedomosti.ru/rss/rubric/it_business",
#     # 'vd_finance': "https://www.vedomosti.ru/rss/rubric/finance",
#     # 'vd_opinion': "https://www.vedomosti.ru/rss/rubric/opinion",
#     # 'vd_analytics': "https://www.vedomosti.ru/rss/rubric/opinion/analytics",


#     # RT
#     # 'rt': "https://russian.rt.com/rss/",
# }


SOURCES = {
    # "rt": "https://russian.rt.com/rss",
    "vd": "https://www.vedomosti.ru/rss/news",

    # for buhgalter
    "buh": "https://buh.ru/rss/?chanel=news",
    "klerk": "https://www.klerk.ru/export/news.rss",
    "audit-it": "http://www.audit-it.ru/rss/news_all.xml",
    # "rb": "https://rb.ru/feeds/all/",

    # for business owner
    # "rb": "https://rb.ru/feeds/all/",
    "rb_pr": "https://rb.ru/feeds/tag/pr/",
    "rb_finance": "https://rb.ru/feeds/tag/fintech/",
    "rb_hr": "https://rb.ru/feeds/tag/hr/",
    "rb_crypto": "https://rb.ru/feeds/tag/crypto/",
    "rb_marketing": "https://rb.ru/feeds/tag/marketing/",
    "rmblr_finance": "https://finance.rambler.ru/rss/economics/",
    "rmblr_business": "https://finance.rambler.ru/rss/business/",
    "rmblr_markets": "https://finance.rambler.ru/rss/markets/",
}

rb_topics = {
    "rb_pr":"PR",
    "rb_finance":"Финансы",
    "rb_hr":"HR",
    "rb_crypto":"Криптовалюты",
    "rb_marketing":"Маркетинг",
}

class RSSParser:
    def __init__(self, sources: dict[str,str]):
        self.sources = sources

    def fetch_entries(self) -> list[dict]:
        entries = []
        for source, url in self.sources.items():
            feed = feedparser.parse(url)
            # If there is no tags for entries, skip source and print warning
            if not feed['entries'][0].get('tags') and source not in rb_topics:
                print(f"Warning: No tags for source {source}")
                continue

            for entry in feed['entries']:
                entry['source'] = source
                entries.append(entry)
        return entries

    def standardize_general(self, entry: dict) -> dict:
        """ Turns entry to a standardized format

        Args:
            entry (dict): entry from feedparser

        Returns:
            dict: standardized entry in a format:
        {
            'source': str,
            'title': str,
            'url':  str,
            'date': timestamp with zone,
            'tags': list[str],
            'text': str,
        }
        """
        entry =  {
            'source': entry['source'],
            'title': entry['title'].replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip(),
            'url':  entry['link'],
            'date': entry['published_parsed'],
            'tags': [tag['term'] for tag in entry['tags']] if 'tags' in entry else [rb_topics[entry['source']]] if entry['source'] in rb_topics else [],
            'text': entry['summary'] if 'summary' in entry else '',
        }
        # pprint(entry)

        return entry

    def get_last_standardized_news(self) -> list[dict]:
        entries = self.fetch_entries()
        return [self.standardize_general(entry) for entry in entries]


if __name__ == '__main__':
    parser = RSSParser(SOURCES)
    from pprint import pprint
    news = parser.get_last_standardized_news()

    # Count every tag occurence
    tags = {}
    for entry in news:
        for tag in entry['tags']:
            if tag not in tags:
                tags[tag] = 0
            tags[tag] += 1

    # Plot tag distribution
    import matplotlib.pyplot as plt
    # Sort by value
    # Set plot size
    plt.rcParams["figure.figsize"] = (20,5)
    tags = {k: v for k, v in sorted(tags.items(), key=lambda item: item[1], reverse=True)}
    plt.bar(tags.keys(), tags.values())


    print(f"Total news: {len(news)}")

    # Print unique tags
    print("Unique tags:")
    print(tags.keys())

    plt.xticks(rotation=90)
    plt.show()


