import re

def split_sentences(text):
    # Split on whitespace that follows a period, question mark, or exclamation mark
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty strings if any
    return [s for s in sentences if s]

if __name__ == '__main__':
    text = "Hello there! How are you? I am fine. This is a test."
    print(split_sentences(text))
