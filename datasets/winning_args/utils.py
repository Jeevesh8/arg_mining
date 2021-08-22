import re

def clean_text(text: str) -> str:
    """
    Performs basic cleaning of text, handles URLs and
    quotes by adding additional tags.
    """
    url_regex = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,4}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"

    text = text.strip(" _\t\n")
    text = text.split("____")[0]  # To remove footnotes
    text = text.strip(" _\t\n")
    text = re.sub(url_regex, "<url>", text)  # To remove URLs
    text = re.sub(r"&gt;.*(?!(\n+))$", "",
                    text)  # To remove quotes at last.
    text = re.sub(r"&gt;(.*)\n", "<startq> \g<1> <endq>",
                    text)  # To add start quote, end quote tags
    text = re.sub(r"\n", " ", text)
    text = text.rstrip(" _\n\t")
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\r", " ", text)
    text = text.lower()
    return text
