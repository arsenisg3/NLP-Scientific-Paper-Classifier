from strenum import StrEnum


class SessionFields(StrEnum):
    ID = "id"
    TITLE = "title"
    CATEGORY = "category"
    FIRST_AUTHOR = "first_author"
    SUMMARY = "summary"
    SUMMARY_WORD_COUNT = "summary_word_count"

    # Normalized
    TITLE_NORM = f"{TITLE}_normalized"
    FIRST_AUTHOR_NORM = f"{FIRST_AUTHOR}_normalized"
    SUMMARY_NORM = f"{SUMMARY}_normalized"
