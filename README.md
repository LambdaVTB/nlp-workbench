# lambda() -  MORE.Tech 4.0 - nlp-workbench

Репозиторий для разработки и апробации различных моделей и подходов к обработке текста. Здесь собраны различные ноутбуки, которые можно использовать как примеры.

<!-- Для функциональной части проекта обратитесь к репозиторию [бекенда](). -->

## Основные репозитории проекта

- [Бекенд, запуск проекта](https://github.com/LambdaVTB/backend-bundle)
- [Фронтенд, мобильный интерфейс](https://github.com/LambdaVTB/flutter-frontend)
- [nlp-workbench, место, где собраны ноутбуки](https://github.com/LambdaVTB/nlp-workbench) <-- Вы тут

## Оглавление

- [lambda() -  MORE.Tech 4.0 - nlp-workbench](#lambda----moretech-40---nlp-workbench)
  - [Основные репозитории проекта](#основные-репозитории-проекта)
  - [Оглавление](#оглавление)
  - [Структура репозитория](#структура-репозитория)
  - [Используемые библиотеки](#используемые-библиотеки)
  - [Разработанные скрипты](#разработанные-скрипты)
    - [scripts/ner_parser.py](#scriptsner_parserpy)
    - [scripts/similarity_model.py](#scriptssimilarity_modelpy)
    - [scripts/rss_parser.py](#scriptsrss_parserpy)
    - [scripts/get_trends.py](#scriptsget_trendspy)
    - [scripts/get_trends_tfidf.py](#scriptsget_trends_tfidfpy)
      - [Алгоритм](#алгоритм)
  - [Остались вопросы?](#остались-вопросы)
  - [lambda() - это мы](#lambda---это-мы)

## Структура репозитория

- `scripts` - папка с классами и скриптами, готовыми к использованию в проде
- `notebooks` - папка с ноутбуками, в которых производилась разработка скриптов и где продемонстрирован их функционал
- `models` - файлы моделей и векторайзеров
- `artifacts` - срезы данных, которые можно использовать для тестирования скриптов

## Используемые библиотеки

- [Проект Natasha](https://natasha.github.io/) - набор Python-библиотек для обработки текстов на естественном русском языке
- [pandas и pandarallel](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) - парсинг HTML
- [requests](https://requests.readthedocs.io/en/master/) - отправка запросов

## Разработанные скрипты

### [scripts/ner_parser.py](scripts/ner_parser.py)

- Скрипт для NER-парсинга (Named Entity Recognition) заголовков. Модифицирован, чтобы так же выделять и простые существительные от других слов. Используется в проде. Используется в get_trends.py.

### [scripts/similarity_model.py](scripts/similarity_model.py)

- Скрипт-попытка определения семантической близости заголовков с помощью нейронных моделей. Не используется в проде.

### [scripts/rss_parser.py](scripts/rss_parser.py)

- Скрипт для парсинга RSS-лент с помощью `feedparser`. Немного адаптирован, чтобы автоматически добавлять тег категории для некоторых источников. Разработан так, чтобы было просто добавить новые источники. Почему RSS? Потому что это самый простой способ получить новости с сайтов, которые не предоставляют API. Используется в get_trends.py.

### [scripts/get_trends.py](scripts/get_trends.py)

- Скрипт для получения топов тем по входящему датафрейму с новостями. При использовании на датафрейме ограниченном за последнее время, получаем тренды. Также реализована функция для получения трендов в рамках
набора тегов. Это позволяет делать персональные тренды для пользователей. В прототипе - используется при персонализации трендов для ролей, а в перспективе можно использовать для персонализации трендов для пользователей на основе их предпочтений по тегам.

### [scripts/get_trends_tfidf.py](scripts/get_trends_tfidf.py)

- то же самое, что и scripts/get_trends.py, однако здесь так же используется TF-IDF. Это замедляет работу скрипта в несколько раз, однако тренды получаются значительно точнее. (см. пример ниже)

![](vis/example_tfidf.png)

#### Алгоритм

0. Если хотим получить тренды по тегам, то фильтруем датафрейм по тегам, то же касатеся временных периодов
1. Для каждой новости парсим заголовок и получаем список сущностей (NER-парсинг), и список существительных (NER-парсинг + регулярки), по обученной TF-IDF на заголовках новостей за последний период определяем самые важные слова, так же добавляем их.
2. Производим лемматизацию, чтобы различные формы слов при подсчете считались одинаковыми.
3. Считаем количество слов и возвращаем в качестве словаря список трендов. По словарю можно построить красивые графики. Стоит так же отметить, что список сущностей включает в себя классификацию сущностей на организации, людей и места, а так же общие темы по TF-IDF. См. более подробные прмеры работы в README_whys.md

## Остались вопросы?

- Дополнительная информация по нашим архитектурным и алгоритмическим решениям представлена в [Почемучках](README_whys.md), а если что-то не работает, то обращайтесь к нам! Мы всегда рады помочь и объяснить.

## lambda() - это мы

- Лебедева Татьяна - мобильный разработчик
- Голубев Егор - бекенд разработчик
- Тампио Илья - ML разработчик
- Денис Лахтионов - дизайнер
- Егор Смурыгин - менеджер
